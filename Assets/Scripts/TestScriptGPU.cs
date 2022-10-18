using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using K4AdotNet.Record;
using K4AdotNet.Sensor;
using TurboJpegWrapper;
using System.Threading.Tasks;

public class TestScriptGPU : MonoBehaviour
{
    [SerializeField]
    ComputeShader computeShader;
    Playback kinectVideo;
    Transformation kinectTransform;
    Calibration kinectCalibration;
    Renderer rendererComponent;
    ComputeBuffer depthBuffer;
    static readonly int
        pixelBufferID = Shader.PropertyToID("pixelBuffer"),
        outputBufferID = Shader.PropertyToID("outputBuffer"),
        depthBufferID = Shader.PropertyToID("depthBuffer"),
        imageWidthID = Shader.PropertyToID("imageWidth"),
        imageHeightID = Shader.PropertyToID("imageHeight");
    RenderTexture rt;
    RenderTexture outputTexture;
    Texture2D tex;
    Texture2D blankBackground;
    short[] colorDepth;
    byte[] imgBuffer;
    TJDecompressor tJDecompressor;
    Image outputImg = new Image(ImageFormat.Depth16, 1280, 720, ImageFormats.StrideBytes(ImageFormat.Depth16, 1280));
    // Start is called before the first frame update
    void Start()
    {
        Physics.autoSimulation = false;
        kinectVideo = new Playback("C:/Users/zhang/OneDrive/Desktop/output.mkv");
        kinectVideo.GetCalibration(out kinectCalibration);
        kinectTransform = new Transformation(kinectCalibration);

        //kinectVideo.TrySeekTimestamp(K4AdotNet.Microseconds64.FromSeconds(5.8), PlaybackSeekOrigin.Begin);
        kinectVideo.TryGetNextCapture(out var capture);
        rendererComponent = GetComponent<Renderer>();
        rt = new RenderTexture(1280, 720, 0);
        rt.enableRandomWrite = true;
        outputTexture = new RenderTexture(1280, 720, 0);
        outputTexture.enableRandomWrite = true;
        RenderTexture.active = outputTexture;
        depthBuffer = new ComputeBuffer(1280 * 720 / 2, 4);
        tex = new Texture2D(1280, 720, TextureFormat.RGBA32, false);
        blankBackground = new Texture2D(1280, 720, TextureFormat.RGBA32, false);
        for (int i = 0; i < 720; i++)
        {
            for (int j = 0; j < 1280; j++)
            {
                blankBackground.SetPixel(j, i, Color.black);
            }
        }
        blankBackground.Apply();
        colorDepth = new short[1280 * 720];
        imgBuffer = new byte[1280 * 720 * 4];
        computeShader.SetInt(imageWidthID, 1280);
        computeShader.SetInt(imageHeightID, 720);
        computeShader.SetBuffer(0, depthBufferID, depthBuffer);
        computeShader.SetTexture(0, pixelBufferID, rt);
        computeShader.SetTexture(0, outputBufferID, outputTexture);
        tJDecompressor = new TJDecompressor();
    }

    private void OnEnable()
    {
        Start();
    }

    private void OnDisable()
    {
        depthBuffer.Release();
        kinectTransform.Dispose();
        kinectVideo.Dispose();
    }

    void UpdateFunctionOnGPU()
    {
        computeShader.Dispatch(0, 1280 / 8, 720 / 8, 1);
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
                            Task.Run(() => kinectTransform.DepthImageToColorCamera(depthImg, outputImg)),
                            Task.Run(() => tJDecompressor.Decompress(img.Buffer, (ulong)img.SizeBytes, imgBufferPtr, img.WidthPixels * img.HeightPixels * 4, TJPixelFormats.TJPF_RGBA, TJFlags.BOTTOMUP))
                        };
                        Task.WaitAll(tasks);
                    }
                }
                outputImg.CopyTo(colorDepth);
                depthBuffer.SetData(colorDepth);
                tex.LoadRawTextureData(imgBuffer);
                tex.Apply();
                Graphics.Blit(tex, rt);
                Graphics.Blit(blankBackground, outputTexture);
                //raw data is in the format of RGBA
                UpdateFunctionOnGPU();
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
}
