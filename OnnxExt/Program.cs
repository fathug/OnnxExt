using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;

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

                //int[] inputShape = inputMeta[inputName].Dimensions.ToArray();
                //float[] inputData = Enumerable.Range(0, inputShape.Aggregate(1, (a, b) => a * b)).Select(x => (float)x).ToArray(); // 创建一些示例数据
                //var inputTensor = new DenseTensor<float>(inputData, inputShape);

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
                        //Console.WriteLine($"输出张量的形状: [{string.Join(",", outputTensor.Dimensions.ToArray())}]");
                        //Console.WriteLine("输出结果的前几个值:");
                        //for (int i = 0; i < Math.Min(10, outputTensor.Length); i++)
                        //{
                        //    Console.Write($"{outputTensor.GetValue(i)} ");
                        //}
                        //Console.WriteLine();

                        ProcessOutput(outputTensor, imagePath);
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

                // 创建输入张量（形状为 [1, 3, 640, 640]）
                return new DenseTensor<float>(inputData, new int[] { 1, 3, InputHeight, InputWidth });
            }
        }

        // 处理模型输出结果的函数 (此部分是简化版，真实的后处理需要更多步骤)
        static void ProcessOutput(DenseTensor<float> outputTensor, string imagePath)
        {
            // 获取输出张量的形状
            int[] outputShape = outputTensor.Dimensions.ToArray();

            Console.WriteLine($"模型输出张量形状: [{string.Join(",", outputShape)}]");

            // 解析输出数据 (这里只是一个简单的示例，实际应用中需要根据模型输出格式进行解析)
            // YOLOv5 输出通常是 (batch_size, num_detections, 85) 或 (batch_size, num_detections, 84)
            // 85 包含 (x, y, w, h, confidence, 80 classes)
            // 84 包含 (x, y, w, h, confidence, 79 classes)，一些模型可能没有 classes 置信度。
            // 这里我们简化输出的处理，只打印检测框置信度高于0.5的检测信息。

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
                if (confidence > 0.05)
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

            Console.WriteLine("如果需要可视化检测框，请自行添加绘制图像的代码");

        }

        public class ObjectResult
        {
            public float Confidence { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float W { get; set; }
            public float H { get; set; }
        }
    }
}