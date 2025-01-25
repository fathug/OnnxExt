/// <summary>
/// 实现推理结果的实时显示。
/// 提供模型选择，视频流和图片推理功能。
/// </summary>
/// <remarks>
/// 2025-01-23 新建。推理结果显示。
/// 2025-01-23 新增。实现视频流推理。
/// 2025-01-25 修复。推理过程中直接关闭form报异常：访问已释放资源，先停止推理再关闭无异常。
/// </remarks>
using AForge.Video;
using AForge.Video.DirectShow;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OnnxExtTest2
{
    public partial class Form1 : Form
    {
        Stopwatch stopwatch = new Stopwatch();

        public static string _modelPath { get; set; }
        public static string _imagePath { get; set; }

        const float Confidence = 0.45F;
        const float Nms = 0.45F;

        public const string HistoryDirectory = @"D:\WVsionData";

        private FilterInfoCollection videoDevices;
        private VideoCaptureDevice videoSource;

        // 全局变量，用于复用对象
        private SessionOptions gpuSessionOptions;
        private InferenceSession session;
        private string inputName;
        private string outputName;

        public static int InputWidth { get; set; }
        public static int InputHeight { get; set; }
        private Bitmap resizedBitmap = null;

        public Form1()
        {
            InitializeComponent();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "C:\\WVsion"; // 默认打开的目录
                openFileDialog.Filter = "*.onnx|*.onnx|所有文件|*.*"; // 文件筛选器

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    _modelPath = openFileDialog.FileName;
                    textBox1.AppendText("选择模型: " + _modelPath + "\r\n");
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "C:\\WVsion"; // 默认打开的目录
                openFileDialog.Filter = "图片|*.jpg;*.jpeg;*.png;*.bmp|所有文件|*.*"; // 文件筛选器

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    _imagePath = openFileDialog.FileName;
                    textBox1.AppendText("选择图片: " + _imagePath + "\r\n");
                }
            }
        }

        private async void button2_Click(object sender, EventArgs e)
        {
            try
            {
                // 加载图片并更新 UI（主线程操作）
                Bitmap image0 = new Bitmap(_imagePath);

                Bitmap image0Copy = new Bitmap(image0);

                pictureBox1.Image = image0;

                // 整个推理过程使用异步方法
                List<ObjectResult> result = await Task.Run(() =>
                {
                    DenseTensor<float> inputTensor = PreprocessImage(image0Copy);
                    var inputs = new List<NamedOnnxValue>()
                    {
                        NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
                    };

                    // 推理
                    using var results = session.Run(inputs);
                    var outputTensor = results.FirstOrDefault(r => r.Name == outputName)?.Value as DenseTensor<float>;

                    if (outputTensor == null)
                        throw new Exception("推理结果为空");

                    // 后处理
                    List<ObjectResult> objectResults = ProcessOutput(outputTensor);
                    List<ObjectResult> objectResultsNms0 = NonMaxSuppression(objectResults, Nms);
                    return objectResultsNms0;
                });

                var image1 = DrawBoundingBoxes(result, image0Copy);

                // 创建副本用于显示，否则会报错资源占用
                Bitmap image1Copy = new Bitmap(image1);

                // 显示到UI，主线程操作
                pictureBox2.Image = image1Copy;

                // 本地保存
                string outputDir = Path.Combine(HistoryDirectory, DateTime.Now.ToString("yyyyMMdd"));
                if (!Directory.Exists(outputDir))
                {
                    Directory.CreateDirectory(outputDir);
                }
                string outputImagePath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(_imagePath)}_dst.png");
                await Task.Run(() => image1.Save(outputImagePath, ImageFormat.Png));
                textBox1.AppendText($"渲染图像已保存至{outputImagePath}" + "\r\n");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"错误:{ex.Message}");
            }
        }

        // 图像转张量
        static DenseTensor<float> PreprocessImage(Bitmap image)
        {
            // 将图像缩放到指定尺寸
            Bitmap resizedImage = LetterboxResize(image, InputWidth, InputHeight);

            int pixelNum = InputWidth * InputHeight;    // 总像素数

            // 用LockBits加速像素访问
            var bitmapData = resizedImage.LockBits(new Rectangle(0, 0, InputWidth, InputHeight), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            try
            {
                // 在堆上分配内存
                float[] inputData = new float[pixelNum * 3];

                unsafe
                {
                    byte* ptr = (byte*)bitmapData.Scan0; // 获取图像数据首地址

                    for (int y = 0; y < InputHeight; y++)
                    {
                        for (int x = 0; x < InputWidth; x++)
                        {
                            int index = ((y * InputWidth) + x);

                            int ptrIndex = (y * bitmapData.Stride) + (x * 3);

                            // 直接从内存读取并归一化，注意 BGR 顺序
                            fixed (float* pInputData = &inputData[0])
                            {
                                pInputData[index + pixelNum * 0] = ptr[ptrIndex + 2] / 255f; // R
                                pInputData[index + pixelNum * 1] = ptr[ptrIndex + 1] / 255f; // G
                                pInputData[index + pixelNum * 2] = ptr[ptrIndex + 0] / 255f; // B
                            }
                        }
                    }
                }
                // 转为张量，形状为[1,3,InputHeight,InputWidth]
                return new DenseTensor<float>(inputData, new int[] { 1, 3, InputHeight, InputWidth });
            }
            finally
            {
                resizedImage.UnlockBits(bitmapData);
                resizedImage.Dispose();
            }
        }

        // Letterbox缩放
        static Bitmap LetterboxResize(Bitmap image, int targetWidth, int targetHeight)
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

        // 处理模型输出结果
        static List<ObjectResult> ProcessOutput(DenseTensor<float> outputTensor)
        {
            // 获取输出张量的形状
            int[] outputShape = outputTensor.Dimensions.ToArray();

            //Console.WriteLine($"本次输出张量形状: [{string.Join(",", outputShape)}]");

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

        // 非极大值抑制（NMS)
        static List<ObjectResult> NonMaxSuppression(List<ObjectResult> boxes, float iouThreshold)
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
                    boxesOfSameClass = boxesOfSameClass.Where(box => CalculateIoU(currentBox, box) <= iouThreshold).ToList();
                }
            }

            return selectedBoxes;
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

        // 结果绘制
        static Bitmap DrawBoundingBoxes(List<ObjectResult> objectResults, Bitmap image)
        {
            if (objectResults.Count > 0)
            {
                // 获取所有检测到的类别
                HashSet<int> uniqueClassIds = new HashSet<int>(objectResults.Select(r => r.ClassId));
                int numClasses = uniqueClassIds.Count;
                // 动态生成颜色
                Color[] classColors = GenerateDistinctColors(numClasses);

                // 获取Letterbox缩放的比例和偏移量
                float ratio = Math.Min((float)InputWidth / image.Width, (float)InputHeight / image.Height);
                int newWidth = (int)(image.Width * ratio);
                int newHeight = (int)(image.Height * ratio);
                int xOffset = (InputWidth - newWidth) / 2;
                int yOffset = (InputHeight - newHeight) / 2;

                using (var graphics = Graphics.FromImage(image))
                {
                    foreach (var result in objectResults)
                    {
                        // 根据 ClassId 获取对应的颜色
                        int colorIndex = uniqueClassIds.ToList().IndexOf(result.ClassId);
                        var pen = new Pen(classColors[colorIndex % classColors.Length], 1);

                        // 输出坐标转换为原图坐标
                        // 转换到 Letterbox 图像上的坐标
                        float x = result.X - xOffset;
                        float y = result.Y - yOffset;
                        float w = result.W;
                        float h = result.H;

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
            }

            return image;
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

        private void button4_Click(object sender, EventArgs e)
        {
            try
            {
                videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);

                if (videoDevices.Count == 0)
                    throw new Exception("没有找到摄像头设备");

                // 可以将摄像头设备添加到 ComboBox 中供用户选择
                // foreach (FilterInfo device in videoDevices)
                // {
                //     comboBox1.Items.Add(device.Name);
                // }
                // comboBox1.SelectedIndex = 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"错误: {ex.Message}");
            }

            if (videoSource == null || !videoSource.IsRunning)
            {
                videoSource = new VideoCaptureDevice(videoDevices[0].MonikerString); // 默认选择第一个摄像头
                videoSource.NewFrame += new NewFrameEventHandler(video_NewFrame);
                videoSource.Start();
            }
        }

        // 摄像头每一帧图像回调
        private async void video_NewFrame(object sender, NewFrameEventArgs eventArgs)
        {
            if (this.IsDisposed || videoSource == null)
            {
                return; // 如果资源已释放，直接返回
            }

            try
            {
                stopwatch.Restart();

                // 将摄像头图像显示到 pictureBoxCam
                Bitmap image0 = (Bitmap)eventArgs.Frame.Clone();

                Bitmap image0Copy = new Bitmap(image0);

                pictureBox1.Image = image0;

                List<ObjectResult> objectResultsNms = await Task.Run(() =>
                {
                    DenseTensor<float> inputTensor = PreprocessImage(image0Copy);

                    var inputs = new List<NamedOnnxValue>()
                    {
                        NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
                    };

                    // 推理
                    using var results = session.Run(inputs);
                    var outputTensor = results.FirstOrDefault(r => r.Name == outputName)?.Value as DenseTensor<float>;

                    if (outputTensor == null)
                        throw new Exception("推理结果为空");

                    // 后处理
                    List<ObjectResult> objectResults = ProcessOutput(outputTensor);
                    List<ObjectResult> objectResultsNms0 = NonMaxSuppression(objectResults, Nms);
                    return objectResultsNms0;
                });

                stopwatch.Stop();

                this.Invoke((MethodInvoker)(() =>
                {
                    textBox1.AppendText($"run cost {stopwatch.ElapsedMilliseconds}ms" + "\r\n");
                }));

                var image1 = DrawBoundingBoxes(objectResultsNms, image0);

                // 创建副本用于显示，否则会报错资源占用
                Bitmap image1Copy = new Bitmap(image1);
                pictureBox2.Image = image1Copy;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"错误:{ex.Message}");
            }
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (videoSource != null && videoSource.IsRunning)
            {
                videoSource.SignalToStop(); //发送停止信号
                videoSource.WaitForStop();  //等待摄像头线程结束
                videoSource.NewFrame -= video_NewFrame; //取消订阅,阻止在关闭期间处理新帧,不再有新的帧数据被处理
                videoSource = null;
            }

            gpuSessionOptions?.Dispose();
            session?.Dispose();
        }

        private void button5_Click(object sender, EventArgs e)
        {
            // 创建InferenceSession
            int gpuDeviceId = 0;
            gpuSessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId);
            session = new InferenceSession(_modelPath);

            var inputMeta = session.InputMetadata;
            var outputMeta = session.OutputMetadata;

            inputName = inputMeta.First().Key;
            outputName = outputMeta.First().Key;

            // 获取输入张量尺寸
            InputWidth = inputMeta.First().Value.Dimensions.ToArray()[2];
            InputHeight = inputMeta.First().Value.Dimensions.ToArray()[3];

            resizedBitmap = new Bitmap(InputWidth, InputHeight);

            textBox1.AppendText($"模型已加载: " + _modelPath + "\r\n");
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (videoSource != null && videoSource.IsRunning)
            {
                videoSource.SignalToStop(); //发送停止信号
                videoSource.WaitForStop();  //等待摄像头线程结束
                videoSource.NewFrame -= video_NewFrame; //取消订阅,阻止在关闭期间处理新帧,不再有新的帧数据被处理
                videoSource = null;
            }

            // 窗口未关闭的时候不能释放session
            //gpuSessionOptions?.Dispose();
            //session?.Dispose();
        }
    }
}
