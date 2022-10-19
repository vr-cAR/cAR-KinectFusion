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
    ComputeBuffer leftDepthBuffer;
    static readonly int
        pixelBufferID = Shader.PropertyToID("pixelBuffer"),
        leftDepthBufferID = Shader.PropertyToID("leftDepthBuffer"),
        outputBufferID = Shader.PropertyToID("outputBuffer"),
        depthBufferID = Shader.PropertyToID("depthBuffer"),
        imageWidthID = Shader.PropertyToID("imageWidth"),
        imageHeightID = Shader.PropertyToID("imageHeight");
    RenderTexture rt;
    RenderTexture outputTexture;
    Texture2D tex;
    Texture2D blankBackground;
    int[] defaultDepthArr;
    short[] colorDepth;
    byte[] imgBuffer;
    TJDecompressor tJDecompressor;
    Image outputImg;
    int DepthKernelID;
    int DrawDepthKernelID;
    int imageWidth;
    int imageHeight;
    // Start is called before the first frame update
    void Start()
    {
        DepthKernelID = computeShader.FindKernel("Depth");
        DrawDepthKernelID = computeShader.FindKernel("DrawDepth");
        Physics.autoSimulation = false;
        kinectVideo = new Playback("C:/Users/zhang/OneDrive/Desktop/output.mkv");
        kinectVideo.GetCalibration(out kinectCalibration);
        kinectTransform = new Transformation(kinectCalibration);
        imageWidth = kinectCalibration.ColorCameraCalibration.ResolutionWidth;
        imageHeight = kinectCalibration.ColorCameraCalibration.ResolutionHeight;

        outputImg = new Image(ImageFormat.Depth16, imageWidth, imageHeight, ImageFormats.StrideBytes(ImageFormat.Depth16, imageWidth));

        //kinectVideo.TrySeekTimestamp(K4AdotNet.Microseconds64.FromSeconds(5.8), PlaybackSeekOrigin.Begin);
        kinectVideo.TryGetNextCapture(out var capture);
        rendererComponent = GetComponent<Renderer>();
        rt = new RenderTexture(imageWidth, imageHeight, 0);
        rt.enableRandomWrite = true;
        outputTexture = new RenderTexture(imageWidth, imageHeight, 0);
        outputTexture.enableRandomWrite = true;
        RenderTexture.active = outputTexture;
        depthBuffer = new ComputeBuffer(imageWidth * imageHeight / 2, 4);
        leftDepthBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
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

        computeShader.SetInt(imageHeightID, imageHeight);
        computeShader.SetInt(imageWidthID, imageWidth);
        computeShader.SetBuffer(DepthKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(DepthKernelID, leftDepthBufferID, leftDepthBuffer);
        computeShader.SetBuffer(DrawDepthKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(DrawDepthKernelID, leftDepthBufferID, leftDepthBuffer);
        computeShader.SetTexture(DrawDepthKernelID, pixelBufferID, rt);
        computeShader.SetTexture(DrawDepthKernelID, outputBufferID, outputTexture);
        tJDecompressor = new TJDecompressor();
        defaultDepthArr = new int[imageHeight * imageWidth];
        System.Array.Fill(defaultDepthArr, 1 << 20);
        Application.targetFrameRate = 60;
    }

    private void OnEnable()
    {
        Start();
    }

    private void OnDisable()
    {
        depthBuffer.Release();
        leftDepthBuffer.Release();
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

                leftDepthBuffer.SetData(defaultDepthArr);
                computeShader.Dispatch(DepthKernelID, imageWidth / 8, imageHeight / 8, 1);
                tex.LoadRawTextureData(imgBuffer);
                tex.Apply();
                Graphics.Blit(tex, rt);
                Graphics.Blit(blankBackground, outputTexture);
                //draw pixels on screen
                computeShader.Dispatch(DrawDepthKernelID, imageWidth / 8, imageHeight / 8, 1);
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
}
