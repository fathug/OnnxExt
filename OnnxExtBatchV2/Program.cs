using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Diagnostics;

namespace OnnxExtBatch
{
    public class Yolov5Detector : IDisposable
    {
        // 常量定义 (配置参数)
        private const int DefaultInputWidth = 160;
        private const int DefaultInputHeight = 160;
        private const float DefaultConfidenceThreshold = 0.45f;
        private const float DefaultNmsThreshold = 0.45f;
        private const int DefaultBatchSize = 200;

        // 成员变量
        // private说明只能在Yolov5Detector类内部访问
        // readonly说明这变量在声明的时候有值了，之后不可修改，这是为了保证关键参数不被修改，本项目中下述参数都是和模型有关，属于不可更改的重要参数
        private readonly InferenceSession _session;
        private readonly int _inputWidth;
        private readonly int _inputHeight;
        private readonly float _confidenceThreshold;
        private readonly float _nmsThreshold;
        private readonly int _batchSize;

        // InputWidth是为了获取_inputWidth值，传递给类之外的代码使用。其实把_inputWidth设为public也可以在外部访问，但是不符合封装的安全性。
        public int InputWidth => _inputWidth;
        public int InputHeight => _inputHeight;

        // 构造函数
        public Yolov5Detector(string modelPath,
                              int inputWidth = DefaultInputWidth,
                              int inputHeight = DefaultInputHeight,
                              float confidenceThreshold = DefaultConfidenceThreshold,
                              float nmsThreshold = DefaultNmsThreshold,
                              int batchSize = DefaultBatchSize)
        {
            // 加载模型
            _session = new InferenceSession(modelPath);

            // 存储配置参数
            _inputWidth = inputWidth;
            _inputHeight = inputHeight;
            _confidenceThreshold = confidenceThreshold;
            _nmsThreshold = nmsThreshold;
            _batchSize = batchSize;

            // 打印模型信息 (可选)
            PrintModelInfo();
        }

        // 公开方法 - 执行推理
        public List<List<ObjectResult>> Detect(List<string> imagePaths)
        {
            // 4.1 图像预处理
            DenseTensor<float> inputTensor = PreprocessImages(imagePaths);

            // 4.2 执行推理
            var inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor(_session.InputMetadata.First().Key, inputTensor)
            };

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs))
            {
                // 4.3 结果后处理
                DenseTensor<float> outputTensor = results.First().Value as DenseTensor<float>;
                List<List<ObjectResult>> objectResults = ProcessOutput(outputTensor);
                List<List<ObjectResult>> objectResultsNms = NonMaxSuppressionBatch(objectResults);

                return objectResultsNms;
            }
        }

        // 私有方法 - 图像预处理
        private DenseTensor<float> PreprocessImages(List<string> imagePaths)
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
        private List<List<ObjectResult>> ProcessOutput(DenseTensor<float> outputTensor)
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

    // ObjectResult 类
    public class ObjectResult
    {
        public float Confidence { get; set; }
        public float CenterX { get; set; }
        public float CenterY { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
        public int ClassId { get; set; }
    }

    // 工具类
    public static class Utils
    {
        // Letterbox 缩放函数
        public static Bitmap LetterboxResize(Bitmap image, int targetWidth, int targetHeight)
        {
            int imageWidth = image.Width;
            int imageHeight = image.Height;

            // 计算缩放比例，选择较小的缩放比例
            float ratioWidth = (float)targetWidth / imageWidth;
            float ratioHeight = (float)targetHeight / imageHeight;
            float ratio = Math.Min(ratioWidth, ratioHeight);

            // 计算缩放后的尺寸
            int newWidth = (int)(imageWidth * ratio);
            int newHeight = (int)(imageHeight * ratio);

            // 将原始图像缩放到新的尺寸
            Bitmap resizedImage = new Bitmap(image, newWidth, newHeight);

            // 创建一个目标尺寸的空白画布，并填充灰色 (114, 114, 114)
            Bitmap paddedImage = new Bitmap(targetWidth, targetHeight);
            using (var graphics = Graphics.FromImage(paddedImage))
            {
                graphics.Clear(Color.FromArgb(114, 114, 114));

                // 计算缩放后的图像在画布上的位置，使其居中
                int xOffset = (targetWidth - newWidth) / 2;
                int yOffset = (targetHeight - newHeight) / 2;

                // 将缩放后的图像绘制到画布上
                graphics.DrawImage(resizedImage, xOffset, yOffset);
            }

            return paddedImage;
        }

        // 计算IoU
        public static float CalculateIoU(ObjectResult boxA, ObjectResult boxB)
        {
            // 左上角坐标
            float xA1 = boxA.CenterX - boxA.Width / 2;
            float yA1 = boxA.CenterY - boxA.Height / 2;
            float xA2 = boxA.CenterX + boxA.Width / 2;
            float yA2 = boxA.CenterY + boxA.Height / 2;
            // 右下角坐标
            float xB1 = boxB.CenterX - boxB.Width / 2;
            float yB1 = boxB.CenterY - boxB.Height / 2;
            float xB2 = boxB.CenterX + boxB.Width / 2;
            float yB2 = boxB.CenterY + boxB.Height / 2;

            // 相交部分的坐标
            float xI1 = Math.Max(xA1, xB1);
            float yI1 = Math.Max(yA1, yB1);
            float xI2 = Math.Min(xA2, xB2);
            float yI2 = Math.Min(yA2, yB2);

            // 计算交集面积
            float interArea = Math.Max(0, xI2 - xI1) * Math.Max(0, yI2 - yI1);
            // 计算并集面积
            float boxAArea = boxA.Width * boxA.Height;
            float boxBArea = boxB.Width * boxB.Height;
            float unionArea = boxAArea + boxBArea - interArea;

            // 计算 IoU
            return interArea / unionArea;
        }

        // 生成不同颜色的函数
        public static Color[] GenerateDistinctColors(int count)
        {
            Color[] colors = new Color[count];
            for (int i = 0; i < count; i++)
            {
                // 使用 HSV 颜色空间生成颜色
                colors[i] = ColorFromHSV(i * (360.0 / count), 0.8, 0.9);
            }
            return colors;
        }

        // HSV 转换为 RGB
        public static Color ColorFromHSV(double hue, double saturation, double value)
        {
            int hi = Convert.ToInt32(Math.Floor(hue / 60)) % 6;
            double f = hue / 60 - Math.Floor(hue / 60);

            value = value * 255;
            int v = Convert.ToInt32(value);
            int p = Convert.ToInt32(value * (1 - saturation));
            int q = Convert.ToInt32(value * (1 - f * saturation));
            int t = Convert.ToInt32(value * (1 - (1 - f) * saturation));

            if (hi == 0)
                return Color.FromArgb(255, v, t, p);
            else if (hi == 1)
                return Color.FromArgb(255, q, v, p);
            else if (hi == 2)
                return Color.FromArgb(255, p, v, t);
            else if (hi == 3)
                return Color.FromArgb(255, p, q, v);
            else if (hi == 4)
                return Color.FromArgb(255, t, p, v);
            else
                return Color.FromArgb(255, v, p, q);
        }

        // 结果绘制
        public static void DrawBoundingBoxes(List<ObjectResult> objectResults, string imagePath, int inputWidth, int inputHeight)
        {
            if (objectResults.Count > 0)
            {
                // 获取所有检测到的类别
                HashSet<int> uniqueClassIds = new HashSet<int>(objectResults.Select(r => r.ClassId));
                int numClasses = uniqueClassIds.Count;
                // 动态生成颜色
                Color[] classColors = GenerateDistinctColors(numClasses);

                using (var image = new Bitmap(imagePath))
                {
                    // 获取Letterbox缩放的比例和偏移量
                    float ratio = Math.Min((float)inputWidth / image.Width, (float)inputHeight / image.Height);
                    int newWidth = (int)(image.Width * ratio);
                    int newHeight = (int)(image.Height * ratio);
                    int xOffset = (inputWidth - newWidth) / 2;
                    int yOffset = (inputHeight - newHeight) / 2;

                    using (var graphics = Graphics.FromImage(image))
                    {
                        foreach (var result in objectResults)
                        {
                            // 根据 ClassId 获取对应的颜色
                            int colorIndex = uniqueClassIds.ToList().IndexOf(result.ClassId);
                            var pen = new Pen(classColors[colorIndex % classColors.Length], 1);

                            // 输出坐标转换为原图坐标

                            // 转换到 Letterbox 图像上的坐标
                            float x = result.CenterX - xOffset;
                            float y = result.CenterY - yOffset;
                            float w = result.Width;
                            float h = result.Height;

                            // 转换到原始图像上的坐标
                            float leftTopX = (x - w / 2) / ratio;
                            float leftTopY = (y - h / 2) / ratio;
                            float width = w / ratio;
                            float height = h / ratio;

                            // 绘制矩形框
                            graphics.DrawRectangle(pen, leftTopX, leftTopY, width, height);

                            // 绘制文本
                            graphics.DrawString($"{result.ClassId},{result.Confidence:F2}", new Font("Arial", 9), new SolidBrush(pen.Color), leftTopX, leftTopY);
                        }
                    }

                    // 保存带有检测框的图像
                    string outputImagePath = Path.Combine(Path.GetDirectoryName(imagePath), $"{Path.GetFileNameWithoutExtension(imagePath)}_output.png");
                    image.Save(outputImagePath, ImageFormat.Png);
                    Console.WriteLine($"渲染图像已保存至: {outputImagePath}");
                }
            }
        }
    }

    class Program
    {
        public const string ConfigDirectory = @"C:\WVsion";

        static void Main(string[] args)
        {
            // 文件路径
            string modelPath = Path.Combine(ConfigDirectory,"model", "best0116bs200.onnx");

            string imageDirectory = Path.Combine(ConfigDirectory, "image2");
            // 获取图片文件夹下的所有图片文件路径
            List<string> imagePaths = Directory.GetFiles(imageDirectory, "*.*", SearchOption.TopDirectoryOnly)
                                             .Where(s => s.EndsWith(".png") || s.EndsWith(".jpg") || s.EndsWith(".jpeg"))
                                             .ToList();
            // 创建 Stopwatch
            Stopwatch stopwatch = new Stopwatch();

            // 使用 using 语句确保资源被正确释放
            using (var detector = new Yolov5Detector(modelPath))
            {
                stopwatch.Start();
                List<List<ObjectResult>> results = detector.Detect(imagePaths);
                stopwatch.Stop();
                Console.WriteLine($"Total cost {stopwatch.ElapsedMilliseconds}ms");

                for (int i = 0; i < results.Count; i++)
                {
                    Utils.DrawBoundingBoxes(results[i], imagePaths[i], detector.InputWidth, detector.InputHeight);
                }
            }

            Console.WriteLine("按任意键退出。");
            Console.ReadKey();
        }
    }
}