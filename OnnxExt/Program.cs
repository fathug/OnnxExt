using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxExt
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. 指定 ONNX 模型文件的路径
            string modelPath = "./assets/yolov5n.onnx";

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
                int[] inputShape = inputMeta[inputName].Dimensions.ToArray();
                float[] inputData = Enumerable.Range(0, inputShape.Aggregate(1, (a, b) => a * b)).Select(x => (float)x).ToArray(); // 创建一些示例数据
                var inputTensor = new DenseTensor<float>(inputData, inputShape);
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
                        Console.WriteLine($"输出张量的形状: [{string.Join(",", outputTensor.Dimensions.ToArray())}]");
                        Console.WriteLine("输出结果的前几个值:");
                        for (int i = 0; i < Math.Min(10, outputTensor.Length); i++)
                        {
                            Console.Write($"{outputTensor.GetValue(i)} ");
                        }
                        Console.WriteLine();
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
    }
}