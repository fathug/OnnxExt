using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.IO;
using System.Linq;

namespace OnnxExtDll
{
    public class Utils
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
                    string outputImagePath = Path.Combine(Path.GetDirectoryName(imagePath), $"{Path.GetFileNameWithoutExtension(imagePath)}_output.jpg");
                    image.Save(outputImagePath, ImageFormat.Png);
                    Console.WriteLine($"渲染图像已保存至: {outputImagePath}");
                }
            }
        }


    }
}
