using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using K4AdotNet.Record;
using K4AdotNet.Sensor;
using TurboJpegWrapper;

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
    // Start is called before the first frame update
    void Start()
    {
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
                Image outputImg = new Image(ImageFormat.Depth16, img.WidthPixels, img.HeightPixels, ImageFormats.StrideBytes(ImageFormat.Depth16, img.WidthPixels));
                //Heavy computation time
                kinectTransform.DepthImageToColorCamera(depthImg, outputImg);
                outputImg.CopyTo(colorDepth);
                depthBuffer.SetData(colorDepth);
                unsafe
                {
                    fixed (byte* ptr = imgBuffer)
                    {
                        //faster jpg decompression but still takes a significant amount of time
                        tJDecompressor.Decompress(img.Buffer, (ulong)img.SizeBytes, new System.IntPtr(ptr), img.WidthPixels * img.HeightPixels * 4, TJPixelFormats.TJPF_RGBA, TJFlags.BOTTOMUP);
                    }
                }
                /*
                img.CopyTo(imgBuffer);
                byte[] tempArr = new byte[img.WidthPixels * img.HeightPixels * 4];
                tex.Reinitialize(2, 2);
                //Heavy computation time
                ImageConversion.LoadImage(tex, imgBuffer);
                */
                tex.LoadRawTextureData(imgBuffer);
                tex.Apply();
                Graphics.Blit(tex, rt);
                Graphics.Blit(blankBackground, outputTexture);
                //raw data is in the format of RGBA
                UpdateFunctionOnGPU();
                rendererComponent.material.mainTexture = outputTexture;
                outputImg.Dispose();
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
