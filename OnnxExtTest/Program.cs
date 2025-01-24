using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using OnnxExtDll;

namespace OnnxExtTest
{
    // 测试DLL

    internal class Program
    {
        public const string ConfigDirectory = @"C:\WVsion";

        static void Main(string[] args)
        {
            // 文件路径
            string modelPath = Path.Combine(ConfigDirectory, "model", "best0116bs200.onnx");

            string imageDirectory = Path.Combine(ConfigDirectory, "image2");
            // 获取图片文件夹下的所有图片文件路径
            List<string> imagePaths = Directory.GetFiles(imageDirectory, "*.*", SearchOption.TopDirectoryOnly)
                                             .Where(s => s.EndsWith(".png"))
                                             .ToList();
            // 创建 Stopwatch
            Stopwatch stopwatch = new Stopwatch();

            // 使用 using 语句确保资源被正确释放
            using (var detector = new Yolov5Detector(modelPath, (float)0.45, (float)0.45)) // 2个阈值参数有默认值，但是仍然可以指定
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
