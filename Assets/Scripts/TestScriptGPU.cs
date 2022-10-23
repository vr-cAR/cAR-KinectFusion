using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using K4AdotNet.Record;
using K4AdotNet.Sensor;
using TurboJpegWrapper;
using System.Threading.Tasks;
using UnityEngine.Profiling;

struct TSDF
{
    float tsdfValue;
    float weight;
    public TSDF(float tsdfValue, float weight)
    {
        this.tsdfValue = tsdfValue;
        this.weight = weight;
    }
}

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
    ComputeBuffer normalBuffer;
    ComputeBuffer vertexBuffer;
    ComputeBuffer tsdfBuffer;
    static readonly int
        pixelBufferID = Shader.PropertyToID("pixelBuffer"),
        leftDepthBufferID = Shader.PropertyToID("leftDepthBuffer"),
        outputBufferID = Shader.PropertyToID("outputBuffer"),
        depthBufferID = Shader.PropertyToID("depthBuffer"),
        normalBufferID = Shader.PropertyToID("normalBuffer"),
        imageWidthID = Shader.PropertyToID("imageWidth"),
        imageHeightID = Shader.PropertyToID("imageHeight"),
        leftEyeTranslationDistanceID = Shader.PropertyToID("leftEyeTranslationDistance"),
        spatialWeightID = Shader.PropertyToID("spatialWeight"),
        rangeWeightID = Shader.PropertyToID("rangeWeight"),
        neighborSizeID = Shader.PropertyToID("neighborSize"),
        cameraMatrixID = Shader.PropertyToID("cameraMatrix"),
        invCameraMatrixID = Shader.PropertyToID("invCameraMatrix"),
        vertexBufferID = Shader.PropertyToID("vertexBuffer"),
        tsdfBufferID = Shader.PropertyToID("tsdfBuffer"),
        truncationDistID = Shader.PropertyToID("truncationDist"),
        voxelSizeID = Shader.PropertyToID("voxelSize");
    RenderTexture rt;
    RenderTexture outputTexture;
    Texture2D tex;
    Texture2D blankBackground;
    int[] defaultDepthArr;
    short[] colorDepth;
    byte[] imgBuffer;
    float[] normBufferArr;
    float[] vertexBufferArr;
    TJDecompressor tJDecompressor;
    Image outputImg;
    int DepthKernelID;
    int DrawDepthKernelID;
    int SmoothKernelID;
    int ComputeNormalsID;
    int TSDFUpdateID;
    int imageWidth;
    int imageHeight;
    public float leftEyeTranslationDistance = 0f;
    public bool isPlayingRecording = true;
    public int renderMode = 1;

    public float spatialWeight = 75;
    public float rangeWeight = 75;
    public float truncationDist = .1f;
    public int neighborhoodSize = 10;
    int voxelSize = 256;
    TSDF[,,] tsdfArr;
    Matrix4x4 cameraMatrix;

    // Start is called before the first frame update
    void Start()
    {
        DepthKernelID = computeShader.FindKernel("Depth");
        DrawDepthKernelID = computeShader.FindKernel("DrawDepth");
        SmoothKernelID = computeShader.FindKernel("Smooth");
        ComputeNormalsID = computeShader.FindKernel("ComputeNormals");
        TSDFUpdateID = computeShader.FindKernel("TSDFUpdate");
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

        normalBuffer = new ComputeBuffer(imageWidth * imageHeight * 3, 4);
        vertexBuffer = new ComputeBuffer(imageWidth * imageHeight * 3, 4);
        tsdfBuffer = new ComputeBuffer(voxelSize * voxelSize * voxelSize, 8);
        normBufferArr = new float[imageWidth * imageHeight * 3];
        vertexBufferArr = new float[imageWidth * imageHeight * 3];
        computeShader.SetInt(imageHeightID, imageHeight);
        computeShader.SetInt(imageWidthID, imageWidth);
        computeShader.SetInt(voxelSizeID, voxelSize);
        computeShader.SetBuffer(DepthKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(DepthKernelID, leftDepthBufferID, leftDepthBuffer);
        computeShader.SetBuffer(DrawDepthKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(DrawDepthKernelID, leftDepthBufferID, leftDepthBuffer);
        computeShader.SetTexture(DrawDepthKernelID, pixelBufferID, rt);
        computeShader.SetTexture(DrawDepthKernelID, outputBufferID, outputTexture);
        computeShader.SetBuffer(SmoothKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(SmoothKernelID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(ComputeNormalsID, normalBufferID, normalBuffer);
        computeShader.SetBuffer(ComputeNormalsID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(TSDFUpdateID, tsdfBufferID, tsdfBuffer);
        computeShader.SetBuffer(TSDFUpdateID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(TSDFUpdateID, vertexBufferID, vertexBuffer);
        tJDecompressor = new TJDecompressor();
        defaultDepthArr = new int[imageHeight * imageWidth];
        System.Array.Fill(defaultDepthArr, 1 << 20);
        Application.targetFrameRate = 60;
        tsdfArr = new TSDF[voxelSize, voxelSize, voxelSize];
        cameraMatrix = new Matrix4x4(new Vector4(1, 0, 0, 128), new Vector4(0, 1, 0, 128), new Vector4(0, 0, 1, 128), new Vector4(0, 0, 0, 1));
    }

    private void OnEnable()
    {
        Start();
    }

    private void OnDisable()
    {
        depthBuffer.Release();
        leftDepthBuffer.Release();
        normalBuffer.Release();
        kinectTransform.Dispose();
        kinectVideo.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        Capture capture = null;
        if (!isPlayingRecording)
        {
            kinectVideo.TryGetPreviousCapture(out capture);
        }
        if (!kinectVideo.TryGetNextCapture(out capture))
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
                if (renderMode == 0)
                {
                    PlaneRender();
                }
                else if (renderMode == 1)
                {
                    KinectFusion();
                    PlaneRender();
                }
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

    void PlaneRender()
    {
        computeShader.SetFloat(leftEyeTranslationDistanceID, leftEyeTranslationDistance);
        tex.LoadRawTextureData(imgBuffer);
        tex.Apply();
        outputImg.CopyTo(colorDepth);
        depthBuffer.SetData(colorDepth);

        leftDepthBuffer.SetData(defaultDepthArr);
        computeShader.Dispatch(DepthKernelID, imageWidth / 8, imageHeight / 8, 1);
        Graphics.Blit(tex, rt);
        Graphics.Blit(blankBackground, outputTexture);
        //draw pixels on screen
        computeShader.Dispatch(DrawDepthKernelID, imageWidth / 8, imageHeight / 8, 1);
        rendererComponent.material.mainTexture = outputTexture;
    }

    void KinectFusion()
    {
        outputImg.CopyTo(colorDepth);
        depthBuffer.SetData(colorDepth);
        //TODO: Implement depth/normal map pyramid
        //Use bilateral filtering on depth
        //cameraMatrix = new Matrix4x4(new Vector4(1, 0, 0, leftEyeTranslationDistance), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(0, 0, 0, 1));
        computeShader.SetFloat(spatialWeightID, spatialWeight);
        computeShader.SetFloat(rangeWeightID, rangeWeight);
        computeShader.SetFloat(truncationDistID, truncationDist);
        computeShader.SetInt(neighborSizeID, neighborhoodSize);
        computeShader.SetMatrix(cameraMatrixID, cameraMatrix);
        computeShader.SetMatrix(invCameraMatrixID, cameraMatrix.inverse);
        computeShader.Dispatch(SmoothKernelID, imageWidth / 8, imageHeight / 8, 1);
        //calculate normals at each point
        computeShader.Dispatch(ComputeNormalsID, imageWidth / 8, imageHeight / 8, 1);
        normalBuffer.GetData(normBufferArr);
        
        for (int i = 0; i < imageHeight; i++)
        {
            for (int j = 0; j < imageWidth; j++)
            {
                imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)(Mathf.Abs(normBufferArr[(i * imageWidth + j) * 3 + 0]) * 255);
                imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)(Mathf.Abs(normBufferArr[(i * imageWidth + j) * 3 + 1]) * 255);
                imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)(Mathf.Abs(normBufferArr[(i * imageWidth + j) * 3 + 2]) * 255);
                imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
            }
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
