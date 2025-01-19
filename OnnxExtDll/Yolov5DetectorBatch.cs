using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;

namespace OnnxExtDll
{
    public class Yolov5Detector : IDisposable
    {
        // 常量定义 (配置参数)
        private const float DefaultConfidenceThreshold = 0.45f;
        private const float DefaultNmsThreshold = 0.45f;

        // 成员变量
        // private说明只能在Yolov5Detector类内部访问
        // readonly说明这变量在声明的时候有值了，之后不可修改，这是为了保证关键参数不被修改，本项目中下述参数都是和模型有关，属于不可更改的重要参数
        private readonly InferenceSession _session;
        private static int _inputWidth; // 输入张量的尺寸
        private static int _inputHeight;
        private readonly float _confidenceThreshold;
        private readonly float _nmsThreshold;
        private readonly int _batchSize;

        // InputWidth是为了获取_inputWidth值，传递给类之外的代码使用。其实把_inputWidth设为public也可以在外部访问，但是不符合封装的安全性。
        public int InputWidth => _inputWidth;
        public int InputHeight => _inputHeight;

        // 构造函数
        public Yolov5Detector(string modelPath,
                              float confidenceThreshold = DefaultConfidenceThreshold,
                              float nmsThreshold = DefaultNmsThreshold)
        {
            // 加载模型
            int gpuDeviceId = 0; // The GPU device ID to execute on
            using var gpuSessionOptoins = SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId);

            _session = new InferenceSession(modelPath, gpuSessionOptoins);

            var inputMeta = _session.InputMetadata;
            var outputMeta = _session.OutputMetadata;


            // 存储配置参数
            _inputWidth = inputMeta.First().Value.Dimensions.ToArray()[2];
            _inputHeight = inputMeta.First().Value.Dimensions.ToArray()[3];
            _confidenceThreshold = confidenceThreshold;
            _nmsThreshold = nmsThreshold;
            _batchSize = inputMeta.First().Value.Dimensions.ToArray()[0];

            // 打印模型信息 (可选)
            PrintModelInfo();
        }

        // 公开方法 - 执行推理
        public List<List<ObjectResult>> Detect(List<string> imagePaths)
        {
            // 4.1 图像预处理
            DenseTensor<float> inputTensor = PreprocessImageBatch(imagePaths);

            // 4.2 执行推理
            var inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor(_session.InputMetadata.First().Key, inputTensor)
            };

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs))
            {
                // 4.3 结果后处理
                DenseTensor<float> outputTensor = results.First().Value as DenseTensor<float>;
                List<List<ObjectResult>> objectResults = ProcessOutputBatch(outputTensor);
                List<List<ObjectResult>> objectResultsNms = NonMaxSuppressionBatch(objectResults);

                return objectResultsNms;
            }
        }

        // 私有方法 - 图像预处理
        private DenseTensor<float> PreprocessImageBatch(List<string> imagePaths)
        {
            int pixelCount = _inputWidth * _inputHeight;
            float[] inputData = new float[_batchSize * pixelCount * 3];

            for (int batchIndex = 0; batchIndex < _batchSize; batchIndex++)
            {
                // 使用安全的索引访问方式
                string imagePath = imagePaths.Count > batchIndex ? imagePaths[batchIndex] : null;

                // 如果图片路径为空，填充默认值（例如黑色图片）
                if (string.IsNullOrEmpty(imagePath))
                {
                    continue; // 跳过当前批次
                }

                using (var image = new Bitmap(imagePath))
                {
                    // 将图像缩放到指定尺寸
                    Bitmap resizedImage = Utils.LetterboxResize(image, _inputWidth, _inputHeight);

                    // 用LockBits加速像素访问
                    var bitmapData = resizedImage.LockBits(new Rectangle(0, 0, _inputWidth, _inputHeight), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

                    try
                    {
                        unsafe
                        {
                            byte* ptr = (byte*)bitmapData.Scan0; // 获取图像数据首地址

                            for (int y = 0; y < _inputHeight; y++)
                            {
                                for (int x = 0; x < _inputWidth; x++)
                                {
                                    int index = ((y * _inputWidth) + x);

                                    int ptrIndex = (y * bitmapData.Stride) + (x * 3);

                                    // 直接从内存读取并归一化，注意 BGR 顺序
                                    fixed (float* pInputData = &inputData[0])
                                    {
                                        pInputData[batchIndex * pixelCount * 3 + index + pixelCount * 0] = ptr[ptrIndex + 2] / 255f; // R
                                        pInputData[batchIndex * pixelCount * 3 + index + pixelCount * 1] = ptr[ptrIndex + 1] / 255f; // G
                                        pInputData[batchIndex * pixelCount * 3 + index + pixelCount * 2] = ptr[ptrIndex + 0] / 255f; // B
                                    }
                                }
                            }
                        }
                    }
                    finally
                    {
                        resizedImage.UnlockBits(bitmapData);
                    }
                }
            }

            // 转为张量，形状为[batchSize,3,InputHeight,InputWidth]
            return new DenseTensor<float>(inputData, new int[] { _batchSize, 3, _inputHeight, _inputWidth });
        }

        // 私有方法 - 结果解析
        private List<List<ObjectResult>> ProcessOutputBatch(DenseTensor<float> outputTensor)
        {
            // 获取输出张量的形状
            int[] outputShape = outputTensor.Dimensions.ToArray();

            Console.WriteLine($"本次输出张量形状: [{string.Join(",", outputShape)}]");

            int numDetections = outputShape[1];
            int numClasses = outputShape[2] - 5; // 5 = (x, y, w, h, confidence)
            int elementsPerDetection = outputShape[2];

            List<List<ObjectResult>> batchResults = new List<List<ObjectResult>>();

            for (int b = 0; b < _batchSize; b++)
            {
                List<ObjectResult> objectResults = new List<ObjectResult>();

                for (int i = 0; i < numDetections; i++)
                {
                    int detectionOffset = b * numDetections * elementsPerDetection + i * elementsPerDetection;

                    float confidence = outputTensor.GetValue(detectionOffset + 4);

                    if (confidence > _confidenceThreshold)
                    {
                        int classId = -1;
                        float maxClassScore = 0;

                        for (int j = 0; j < numClasses; j++)
                        {
                            float classScore = outputTensor.GetValue(detectionOffset + 5 + j);
                            if (classScore > maxClassScore)
                            {
                                maxClassScore = classScore;
                                classId = j;
                            }
                        }

                        float centerX = outputTensor.GetValue(detectionOffset + 0);
                        float centerY = outputTensor.GetValue(detectionOffset + 1);
                        float width = outputTensor.GetValue(detectionOffset + 2);
                        float height = outputTensor.GetValue(detectionOffset + 3);

                        var result = new ObjectResult()
                        {
                            Confidence = confidence,
                            CenterX = centerX,
                            CenterY = centerY,
                            Width = width,
                            Height = height,
                            ClassId = classId,
                        };
                        objectResults.Add(result);
                    }
                }

                batchResults.Add(objectResults);
            }
            return batchResults;
        }

        // 私有方法 - 批量 NMS
        private List<List<ObjectResult>> NonMaxSuppressionBatch(List<List<ObjectResult>> boxesBatch)
        {
            List<List<ObjectResult>> nmsBatchedResults = new List<List<ObjectResult>>();
            foreach (var boxes in boxesBatch)
            {
                nmsBatchedResults.Add(NonMaxSuppression(boxes));
            }
            return nmsBatchedResults;
        }

        // 私有方法 - 单张 NMS
        private List<ObjectResult> NonMaxSuppression(List<ObjectResult> boxes)
        {
            // 根据置信度对框进行降序排序
            boxes.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

            List<ObjectResult> selectedBoxes = new List<ObjectResult>();

            // 获取所有检测到的类别
            HashSet<int> uniqueClassIds = new HashSet<int>(boxes.Select(r => r.ClassId));

            // 对每个类别分别执行 NMS
            foreach (int classId in uniqueClassIds)
            {
                List<ObjectResult> boxesOfSameClass = boxes.Where(box => box.ClassId == classId).ToList();

                while (boxesOfSameClass.Count > 0)
                {
                    // 选择当前置信度最高的框
                    ObjectResult currentBox = boxesOfSameClass[0];
                    selectedBoxes.Add(currentBox);
                    boxesOfSameClass.RemoveAt(0);

                    // 遍历剩余的框，如果与当前框的 IoU 大于阈值，则移除
                    boxesOfSameClass = boxesOfSameClass.Where(box => Utils.CalculateIoU(currentBox, box) <= _nmsThreshold).ToList();
                }
            }

            return selectedBoxes;
        }

        // 打印模型信息 (可选)
        private void PrintModelInfo()
        {
            var inputMeta = _session.InputMetadata;
            var outputMeta = _session.OutputMetadata;

            Console.WriteLine("模型输入:");
            foreach (var item in inputMeta)
            {
                Console.WriteLine($"  Name: {item.Key}, Type: {item.Value.ElementType}, Shape: [{string.Join(",", item.Value.Dimensions)}]");
            }

            Console.WriteLine("模型输出:");
            foreach (var item in outputMeta)
            {
                Console.WriteLine($"  Name: {item.Key}, Type: {item.Value.ElementType}, Shape: [{string.Join(",", item.Value.Dimensions)}]");
            }
        }

        // 10. 释放资源
        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    public class ObjectResult
    {
        public float Confidence { get; set; }
        public float CenterX { get; set; }
        public float CenterY { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
        public int ClassId { get; set; }
    }
}
