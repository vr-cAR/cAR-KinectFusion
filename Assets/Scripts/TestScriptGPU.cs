using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using K4AdotNet.Record;
using K4AdotNet.Sensor;
using TurboJpegWrapper;
using System.Threading.Tasks;
using UnityEngine.Profiling;

public class TestScriptGPU : MonoBehaviour
{
    [SerializeField]
    ComputeShader computeShader;
    Playback kinectVideo;
    Transformation kinectTransform;
    Calibration kinectCalibration;
    Renderer rendererComponent;
    ComputeBuffer depthBuffer;
    ComputeBuffer translationMapBuffer;
    static readonly int
        pixelBufferID = Shader.PropertyToID("pixelBuffer"),
        translationMapBufferID = Shader.PropertyToID("translationMapBuffer"),
        outputBufferID = Shader.PropertyToID("outputBuffer"),
        depthBufferID = Shader.PropertyToID("depthBuffer"),
        imageWidthID = Shader.PropertyToID("imageWidth"),
        imageHeightID = Shader.PropertyToID("imageHeight");
    RenderTexture rt;
    RenderTexture outputTexture;
    Texture2D tex;
    Texture2D blankBackground;
    short[] blankDepth;
    short[] colorDepth;
    byte[] imgBuffer;
    int[] translationMapBufferArr;
    TJDecompressor tJDecompressor;
    Image outputImg = new Image(ImageFormat.Depth16, 1280, 720, ImageFormats.StrideBytes(ImageFormat.Depth16, 1280));
    int CSMainKernelID;
    int DrawKernelID;
    int imageWidth;
    int imageHeight;
    int[] mapArr;
    // Start is called before the first frame update
    void Start()
    {
        CSMainKernelID = computeShader.FindKernel("CSMain");
        DrawKernelID = computeShader.FindKernel("Draw");
        Physics.autoSimulation = false;
        kinectVideo = new Playback("C:/Users/zhang/OneDrive/Desktop/output.mkv");
        kinectVideo.GetCalibration(out kinectCalibration);
        kinectTransform = new Transformation(kinectCalibration);
        imageWidth = kinectCalibration.ColorCameraCalibration.ResolutionWidth;
        imageHeight = kinectCalibration.ColorCameraCalibration.ResolutionHeight;

        //kinectVideo.TrySeekTimestamp(K4AdotNet.Microseconds64.FromSeconds(5.8), PlaybackSeekOrigin.Begin);
        kinectVideo.TryGetNextCapture(out var capture);
        rendererComponent = GetComponent<Renderer>();
        rt = new RenderTexture(imageWidth, imageHeight, 0);
        rt.enableRandomWrite = true;
        outputTexture = new RenderTexture(imageWidth, imageHeight, 0);
        outputTexture.enableRandomWrite = true;
        RenderTexture.active = outputTexture;
        depthBuffer = new ComputeBuffer(imageWidth * imageHeight / 2, 4);
        translationMapBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
        tex = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);
        blankBackground = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);
        for (int i = 0; i < imageHeight; i++)
        {
            for (int j = 0; j < imageWidth; j++)
            {
                blankBackground.SetPixel(j, i, Color.black);
            }
        }
        blankBackground.Apply();
        colorDepth = new short[imageWidth * imageHeight];
        imgBuffer = new byte[imageWidth * imageHeight * 4];
        translationMapBufferArr = new int[imageWidth * imageHeight];
        computeShader.SetInt(imageWidthID, imageWidth);
        computeShader.SetInt(imageHeightID, imageHeight);
        computeShader.SetBuffer(CSMainKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(CSMainKernelID, translationMapBufferID, translationMapBuffer);
        computeShader.SetBuffer(DrawKernelID, translationMapBufferID, translationMapBuffer);
        computeShader.SetTexture(DrawKernelID, pixelBufferID, rt);
        computeShader.SetTexture(DrawKernelID, outputBufferID, outputTexture);
        tJDecompressor = new TJDecompressor();
        mapArr = new int[imageHeight * imageWidth];
        Application.targetFrameRate = 60;
    }

    private void OnEnable()
    {
        Start();
    }

    private void OnDisable()
    {
        depthBuffer.Release();
        translationMapBuffer.Release();
        kinectTransform.Dispose();
        kinectVideo.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        if (!kinectVideo.TryGetNextCapture(out var capture))
        {
            kinectVideo.TrySeekTimestamp(K4AdotNet.Microseconds64.Zero, PlaybackSeekOrigin.Begin);
            kinectVideo.TryGetNextCapture(out capture);
        }

        if (capture != null)
        {
            Image img = capture.ColorImage;
            Image depthImg = capture.DepthImage;
            if (img != null && depthImg != null)
            {
                unsafe
                {
                    fixed (byte* ptr = imgBuffer)
                    {
                        System.IntPtr imgBufferPtr = new System.IntPtr(ptr);
                        //Run heavy CPU computations concurrently
                        var tasks = new[]
                        {
                            Task.Run(() => DepthToColorTransformationProfiler(depthImg)),
                            Task.Run(() => JPGDecompressProfiler(img, imgBufferPtr))
                        };
                        Task.WaitAll(tasks);
                    }
                }
                outputImg.CopyTo(colorDepth);
                depthBuffer.SetData(colorDepth);
                //compute mapping between pixels in image to where they will be translated
                computeShader.Dispatch(CSMainKernelID, imageWidth / 8, imageHeight / 8, 1);
                //waits for GPU to finish calculations, so ends up blocking for a significant amount of time. Maybe find out how to do the inverse mapping in the compute shader to avoid this.
                translationMapBuffer.GetData(translationMapBufferArr);
                //Find the inverse map from translated pixels to original pixels where any collision will prioritize pixels that are closer.
                Profiler.BeginSample("Inverse map");
                System.Array.Fill(mapArr, imageHeight * imageWidth);
                List<Task> list = new List<Task>();
                int numTasks = 8;
                Task[] taskArr = new Task[numTasks];
                for (int i = 0; i < numTasks; i++)
                {
                    int index = i;
                    taskArr[i] = Task.Factory.StartNew(() => InverseMapProfiler(index, translationMapBufferArr.Length / numTasks));
                }
                Task.WaitAll(taskArr);
                Profiler.EndSample();
                translationMapBuffer.SetData(mapArr);
                tex.LoadRawTextureData(imgBuffer);
                tex.Apply();
                Graphics.Blit(tex, rt);
                Graphics.Blit(blankBackground, outputTexture);
                //draw pixels on screen
                computeShader.Dispatch(DrawKernelID, imageWidth / 8, imageHeight / 8, 1);
                rendererComponent.material.mainTexture = outputTexture;
            }
            if (img != null)
            {
                img.Dispose();
            }
            if (depthImg != null)
            {
                depthImg.Dispose();
            }
            capture.Dispose();
        }
    }

    void DepthToColorTransformationProfiler(Image depthImg)
    {
        Profiler.BeginThreadProfiling("updateThreads", "DepthToColorTransformationProfiler Task " + Task.CurrentId);
        kinectTransform.DepthImageToColorCamera(depthImg, outputImg);
        Profiler.EndThreadProfiling();
    }

    void JPGDecompressProfiler(Image img, System.IntPtr imgBufferPtr)
    {
        Profiler.BeginThreadProfiling("updateThreads", "JPGDecompressProfiler Task " + Task.CurrentId);
        tJDecompressor.Decompress(img.Buffer, (ulong)img.SizeBytes, imgBufferPtr, img.WidthPixels * img.HeightPixels * 4, TJPixelFormats.TJPF_RGBA, TJFlags.BOTTOMUP);
        Profiler.EndThreadProfiling();
    }

    void InverseMapProfiler(int segment, int segmentSize)
    {
        Profiler.BeginThreadProfiling("updateThreads", "InverseMapProfiler Task " + Task.CurrentId);
        for (int i = 0; i < segmentSize; i++)
        {
            int index = segment * segmentSize + i;
            if (translationMapBufferArr[index] == imageHeight * imageWidth)
            {
                continue;
            }
            if (mapArr[translationMapBufferArr[index]] == imageHeight * imageWidth)
            {
                mapArr[translationMapBufferArr[index]] = index;
            }
            else
            {
                if (colorDepth[index] < colorDepth[mapArr[translationMapBufferArr[index]]])
                {
                    mapArr[translationMapBufferArr[index]] = index;
                }
            }
        }
        Profiler.EndThreadProfiling();
    }
}
