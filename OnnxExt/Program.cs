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

        static void Main(string[] args)
        {
            // 1. 指定 ONNX 模型文件的路径
            string modelPath = "./assets/best_opset13.onnx";
            string imagePath = "./assets/img2.png";

            // 2. 创建 InferenceSession 对象
            using (var session = new InferenceSession(modelPath))
            {
                // 3. 获取模型的输入和输出信息 (可选，但推荐)
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

                // 4. 准备输入数据 (这里以一个简单的浮点型向量为例)
                string inputName = inputMeta.First().Key; // 获取第一个输入的名称

                DenseTensor<float> inputTensor = PreprocessImage(imagePath);

                var inputs = new List<NamedOnnxValue>()
                {
                    NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
                };

                // 5. 运行推理
                using (var results = session.Run(inputs))
                {
                    // 6. 获取输出结果
                    string outputName = outputMeta.First().Key; // 获取第一个输出的名称
                    var outputTensor = results.FirstOrDefault(r => r.Name == outputName)?.Value as DenseTensor<float>;

                    if (outputTensor != null)
                    {
                        List<ObjectResult> objectResults =  ProcessOutput(outputTensor, imagePath);
                        DrawBoundingBoxes(objectResults, imagePath);
                    }
                    else
                    {
                        Console.WriteLine("未找到输出结果或输出类型不匹配。");
                    }
                }
            }

            Console.WriteLine("推理完成，按任意键退出。");
            Console.ReadKey();
        }

        
        #region 数据预处理，使用bitmap

        // 图像预处理函数
        static DenseTensor<float> PreprocessImage(string imagePath)
        {
            // 读取图像
            using (var image = new Bitmap(imagePath))
            {
                // 将图像缩放到指定尺寸
                Bitmap resizedImage = new Bitmap(image, InputWidth, InputHeight);

                // 将图像数据转换为浮点数组，并进行归一化和通道顺序调整
                float[] inputData = new float[InputWidth * InputHeight * 3];
                /* 
                 * 注释为错误转换方式
                for (int y = 0; y < InputHeight; y++)
                {
                    for (int x = 0; x < InputWidth; x++)
                    {
                        Color pixel = resizedImage.GetPixel(x, y);
                        inputData[(y * InputWidth + x) * 3 + 0] = pixel.R / 255f; // R
                        inputData[(y * InputWidth + x) * 3 + 1] = pixel.G / 255f; // G
                        inputData[(y * InputWidth + x) * 3 + 2] = pixel.B / 255f; // B
                    }
                }
                */

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

            Console.WriteLine($"模型输出张量形状: [{string.Join(",", outputShape)}]");

            //假设批次大小是 1.
            int numDetections = outputShape[1];
            int numClasses = outputShape[2] - 5;
            int lengthD4 = outputShape[2];

            List<ObjectResult> objectResults = new List<ObjectResult>();

            Console.WriteLine("检测结果:");
            for (int i = 0; i < numDetections; i++)
            {
                int idxConfidence = lengthD4 * i + 4; // 每个检测结果中的置信度索引
                float confidence = outputTensor.GetValue(idxConfidence);
                if (confidence > 0.7)
                {
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
                        H = h
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
        }

        /// <summary>
        /// 结果绘制
        /// </summary>
        /// <param name="objectResults"></param>
        /// <param name="imagePath"></param>
        static void DrawBoundingBoxes(List<ObjectResult> objectResults, string imagePath)
        {
            if (objectResults.Count > 0)
            {
                using (var image = new Bitmap(imagePath))
                {
                    using (var graphics = Graphics.FromImage(image))
                    {
                        var pen = new Pen(Color.Red, 1); // 设置画笔颜色和宽度
                        foreach (var result in objectResults)
                        {
                            // 将坐标转换为图像上的实际坐标
                            float xMin = (result.X - result.W / 2) * image.Width / InputWidth;
                            float yMin = (result.Y - result.H / 2) * image.Height / InputHeight;
                            float width = result.W * image.Width / InputWidth;
                            float height = result.H * image.Height / InputHeight;

                            // 绘制矩形框
                            graphics.DrawRectangle(pen, xMin, yMin, width, height);

                            // 绘制标签 (这里简化为只显示置信度)
                            graphics.DrawString($"{result.Confidence:F3}", SystemFonts.DefaultFont, Brushes.Red, xMin, yMin);
                        }

                        /*
                        var pen = new Pen(Color.Green, 1);

                        foreach (var result in objectResults)
                        {
                            // 将坐标和尺寸转换回原始图像的比例
                            float originalWidth = image.Width;
                            float originalHeight = image.Height;
                            float x = result.X / InputWidth * originalWidth;
                            float y = result.Y / InputHeight * originalHeight;
                            float width = result.W / InputWidth * originalWidth;
                            float height = result.H / InputHeight * originalHeight;

                            // 转换坐标系为左上角开始
                            float left = x - width / 2;
                            float top = y - height / 2;

                            // 绘制矩形框
                            graphics.DrawRectangle(pen, left, top, width, height);

                            // 绘制置信度文本 (可选)
                            graphics.DrawString($"{result.Confidence:F2}", new Font("Arial", 16), Brushes.Red, new PointF(left, top));
                        }
                        */
                    }

                    // 保存带有检测框的图像
                    string outputImagePath = Path.Combine(Path.GetDirectoryName(imagePath), $"{Path.GetFileNameWithoutExtension(imagePath)}_output.png");
                    image.Save(outputImagePath, ImageFormat.Png);
                    Console.WriteLine($"带有检测框的图像已保存至: {outputImagePath}");
                }
            }
        }
    }
}