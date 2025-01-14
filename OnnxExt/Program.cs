using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace OnnxExt
{
    class Program
    {
        const int InputWidth = 160;
        const int InputHeight = 160;
        const float Confidence = 0.45F;
        const float Nms = 0.45F;

        static void Main(string[] args)
        {
            // 文件路径
            string modelPath = "./assets/best0113.onnx";
            string imagePath = "./assets/24-5-16_08.27.44_782.png";

            // 创建InferenceSession对象
            using (var session = new InferenceSession(modelPath))
            {
                // 获取模型的输入和输出信息
                var inputMeta = session.InputMetadata;
                var outputMeta = session.OutputMetadata;

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

                // 准备输入数据
                string inputName = inputMeta.First().Key; // 获取第一个输入的名称

                DenseTensor<float> inputTensor = PreprocessImage(imagePath);

                var inputs = new List<NamedOnnxValue>()
                {
                    NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
                };

                // 运行推理
                using (var results = session.Run(inputs))
                {
                    // 获取输出结果
                    string outputName = outputMeta.First().Key; // 获取第一个输出的名称
                    var outputTensor = results.FirstOrDefault(r => r.Name == outputName)?.Value as DenseTensor<float>;

                    if (outputTensor != null)
                    {
                        List<ObjectResult> objectResults =  ProcessOutput(outputTensor, imagePath);

                        List<ObjectResult> objectResultsNms = NonMaxSuppression(objectResults, Nms);

                        DrawBoundingBoxes(objectResultsNms, imagePath);
                    }
                    else
                    {
                        Console.WriteLine("未找到输出结果或输出类型不匹配。");
                    }
                }
            }

            Console.WriteLine("按任意键退出。");
            Console.ReadKey();
        }

        
        #region 数据预处理，使用bitmap

        // 图像预处理函数
        static DenseTensor<float> PreprocessImage(string imagePath)
        {
            using (var image = new Bitmap(imagePath))
            {
                // 将图像缩放到指定尺寸
                Bitmap resizedImage = new Bitmap(image, InputWidth, InputHeight);

                // 将图像数据转换为浮点数组，并进行归一化和通道顺序调整
                float[] inputData = new float[InputWidth * InputHeight * 3];

                for (int y = 0; y < InputHeight; y++)
                {
                    for (int x = 0; x < InputWidth; x++)
                    {
                        Color pixel = resizedImage.GetPixel(x, y);
                        inputData[(y * InputWidth + x) + InputWidth * InputHeight * 0] = pixel.R / 255f; // R
                    }
                }
                for (int y = 0; y < InputHeight; y++)
                {
                    for (int x = 0; x < InputWidth; x++)
                    {
                        Color pixel = resizedImage.GetPixel(x, y);
                        inputData[(y * InputWidth + x) + InputWidth * InputHeight * 1] = pixel.G / 255f; // R
                    }
                }
                for (int y = 0; y < InputHeight; y++)
                {
                    for (int x = 0; x < InputWidth; x++)
                    {
                        Color pixel = resizedImage.GetPixel(x, y);
                        inputData[(y * InputWidth + x) + InputWidth * InputHeight * 2] = pixel.B / 255f; // R
                    }
                }

                // 创建输入张量（形状为 [1, 3, 640, 640]）
                return new DenseTensor<float>(inputData, new int[] { 1, 3, InputHeight, InputWidth });
            }
        }

        #endregion

        // 处理模型输出结果的函数
        static List<ObjectResult> ProcessOutput(DenseTensor<float> outputTensor, string imagePath)
        {
            // 获取输出张量的形状
            int[] outputShape = outputTensor.Dimensions.ToArray();

            Console.WriteLine($"本次输出张量形状: [{string.Join(",", outputShape)}]");

            //假设批次大小是 1.
            int numDetections = outputShape[1];
            int numClasses = outputShape[2] - 5;
            int lengthD4 = outputShape[2];

            List<ObjectResult> objectResults = new List<ObjectResult>();

            for (int i = 0; i < numDetections; i++)
            {
                int idxConfidence = lengthD4 * i + 4; // 每个检测结果中的置信度索引
                float confidence = outputTensor.GetValue(idxConfidence);

                if (confidence > Confidence)
                {
                    int maxClassId = -1;
                    float maxClassScore = 0;

                    int idxStart = idxConfidence + 1;
                    int idxEdn = (i + 1) * lengthD4;

                    for (int j = idxStart; j < idxEdn; j++)
                    {
                        float tmpClasaScore = outputTensor.GetValue(j);
                        if (tmpClasaScore > maxClassScore)
                        {
                            maxClassScore = tmpClasaScore;
                            maxClassId = j - idxConfidence - 1;//这里可能有问题
                        }
                    }

                    float x = outputTensor.GetValue(idxConfidence - 4);
                    float y = outputTensor.GetValue(idxConfidence - 3);
                    float w = outputTensor.GetValue(idxConfidence - 2);
                    float h = outputTensor.GetValue(idxConfidence - 1);

                    var tmp = new ObjectResult()
                    {
                        Confidence = confidence,
                        X = x,
                        Y = y,
                        W = w,
                        H = h,
                        ClassId = maxClassId,
                    };
                    objectResults.Add(tmp);
                }
            }

            return objectResults;
        }

        public class ObjectResult
        {
            public float Confidence { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float W { get; set; }
            public float H { get; set; }
            public int ClassId { get; set; }
        }

        // 结果绘制
        static void DrawBoundingBoxes(List<ObjectResult> objectResults, string imagePath)
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
                    using (var graphics = Graphics.FromImage(image))
                    {
                        foreach (var result in objectResults)
                        {
                            // 根据 ClassId 获取对应的颜色
                            int colorIndex = uniqueClassIds.ToList().IndexOf(result.ClassId);
                            var pen = new Pen(classColors[colorIndex % classColors.Length], 1);

                            // 输出坐标转换为原图坐标
                            float leftTopX = (result.X - result.W / 2) / InputWidth * image.Width;
                            float leftTopY = (result.Y - result.H / 2) / InputHeight * image.Height;
                            float width = result.W * image.Width / InputWidth;
                            float height = result.H * image.Height / InputHeight;

                            // 绘制矩形框
                            graphics.DrawRectangle(pen, leftTopX, leftTopY, width, height);

                            // 绘制文本
                            graphics.DrawString($"{result.ClassId},{result.Confidence:F2}", SystemFonts.DefaultFont, Brushes.Red, leftTopX, leftTopY);
                        }
                    }

                    // 保存带有检测框的图像
                    string outputImagePath = Path.Combine(Path.GetDirectoryName(imagePath), $"{Path.GetFileNameWithoutExtension(imagePath)}_output.png");
                    image.Save(outputImagePath, ImageFormat.Png);
                    Console.WriteLine($"渲染图像已保存至: {outputImagePath}");
                }
            }
        }

        // 计算IoU
        static float CalculateIoU(ObjectResult boxA, ObjectResult boxB)
        {
            // 左上角坐标
            float xA1 = boxA.X - boxA.W / 2;
            float yA1 = boxA.Y - boxA.H / 2;
            float xA2 = boxA.X + boxA.W / 2;
            float yA2 = boxA.Y + boxA.H / 2;
            // 右下角坐标
            float xB1 = boxB.X - boxB.W / 2;
            float yB1 = boxB.Y - boxB.H / 2;
            float xB2 = boxB.X + boxB.W / 2;
            float yB2 = boxB.Y + boxB.H / 2;

            // 相交部分的坐标
            float xI1 = Math.Max(xA1, xB1);
            float yI1 = Math.Max(yA1, yB1);
            float xI2 = Math.Min(xA2, xB2);
            float yI2 = Math.Min(yA2, yB2);

            // 计算交集面积
            float interArea = Math.Max(0, xI2 - xI1) * Math.Max(0, yI2 - yI1);
            // 计算并集面积
            float boxAArea = boxA.W * boxA.H;
            float boxBArea = boxB.W * boxB.H;
            float unionArea = boxAArea + boxBArea - interArea;

            // 计算 IoU
            return interArea / unionArea;
        }

        // 非极大值抑制（NMS)
        static List<ObjectResult> NonMaxSuppression(List<ObjectResult> boxes, float iouThreshold)
        {
            // 根据置信度对框进行降序排序
            boxes.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

            List<ObjectResult> selectedBoxes = new List<ObjectResult>();

            while (boxes.Count > 0)
            {
                // 选择当前置信度最高的框
                ObjectResult currentBox = boxes[0];
                selectedBoxes.Add(currentBox);
                boxes.RemoveAt(0);

                // 遍历剩余的框，如果与当前框的 IoU 大于阈值，则移除
                boxes = boxes.Where(box => CalculateIoU(currentBox, box) <= iouThreshold).ToList();
            }

            return selectedBoxes;
        }

        // 生成不同颜色的函数
        static Color[] GenerateDistinctColors(int count)
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
        static Color ColorFromHSV(double hue, double saturation, double value)
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
    }
}