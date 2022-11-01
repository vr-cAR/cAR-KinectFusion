using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using K4AdotNet.Record;
using K4AdotNet.Sensor;
using TurboJpegWrapper;
using System.Threading.Tasks;
using UnityEngine.Profiling;
using Emgu.CV;
using Emgu.CV.PpfMatch3d;
using System.IO;
struct TSDF
{
    public float tsdfValue;
    public float weight;
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
    ComputeBuffer smoothDepthBuffer;
    ComputeBuffer normalMapBuffer;
    ComputeBuffer vertexMapBuffer;
    ComputeBuffer ICPBuffer;
    ComputeBuffer ICPReductionBuffer;
    ComputeBuffer pointCloudBuffer;
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
        tsdfBufferID = Shader.PropertyToID("TSDFBuffer"),
        truncationDistID = Shader.PropertyToID("truncationDist"),
        voxelSizeID = Shader.PropertyToID("voxelSize"),
        roomSizeID = Shader.PropertyToID("roomSize"),
        cameraScaleID = Shader.PropertyToID("cameraScale"),
        colorIntrinsicMatrixID = Shader.PropertyToID("colorIntrinsicMatrix"),
        invColorIntrinsicMatrixID = Shader.PropertyToID("invColorIntrinsicMatrix"),
        rayTraceStepsID = Shader.PropertyToID("rayTraceSteps"),
        smoothDepthBufferID = Shader.PropertyToID("smoothDepthBuffer"),
        normalMapBufferID = Shader.PropertyToID("normalMapBuffer"),
        vertexMapBufferID = Shader.PropertyToID("vertexMapBuffer"),
        ICPBufferID = Shader.PropertyToID("ICPBuffer"),
        ICPReductionBufferID = Shader.PropertyToID("ICPReductionBuffer"),
        currentICPCameraMatrixID = Shader.PropertyToID("currentICPCameraMatrix"),
        ICPThresholdDistanceID = Shader.PropertyToID("ICPThresholdDistance"),
        ICPThresholdRotationID = Shader.PropertyToID("ICPThresholdRotation"),
        pointCloudBufferID = Shader.PropertyToID("pointCloudBuffer");
    RenderTexture rt;
    RenderTexture outputTexture;
    Texture2D tex;
    Texture2D blankBackground;
    int[] defaultDepthArr;
    byte[] imgBuffer;
    Vector3[,] normBufferArr;
    Vector3[,] vertexBufferArr;
    float[,] smoothDepthOne;
    short[,] smoothDepthTwo;
    short[,] smoothDepthThree;
    TJDecompressor tJDecompressor;
    Image outputImg;
    int FormatDepthBufferID;
    int DepthKernelID;
    int DrawDepthKernelID;
    int SmoothKernelID;
    int ComputeNormalsID;
    int TSDFUpdateID;
    int RenderTSDFID;
    int ICPKernelID;
    int ICPReductionKernelID;
    int imageWidth;
    int imageHeight;
    public float leftEyeTranslationDistance = 0f;
    public bool isPlayingRecording = true;
    public int renderMode = 1;

    public float spatialWeight = 75;
    public float rangeWeight = 75;
    public float truncationDist = 100f;
    public int neighborhoodSize = 10;
    public float roomSize = 5;
    public float cameraScale = 1;
    public int rayTraceSteps = 300;
    public float thresholdDistance = 5f;
    public float thresholdRotation = .1f;
    int voxelSize = 256;
    TSDF[,,] tsdfArr;
    Matrix4x4 cameraMatrix;
    Matrix4x4 colorIntrinsicMatrix;

    Vector3[,] normalMapArr;
    Vector3[,] vertexMapArr;

    int frame = 0;
    bool isTracking;
    bool isFirst;
    public float pixelThreshold = 0;

    public int split = 0;

    float[,] leftMatArr;
    float[] rightValArr;

    float[] ICPReductionBufferArr;
    float[] ICPReductionTotArr;
    float[] ICPReductionResultArr;

    int waveGroupSize = 256 * 64;

    short[] pointCloudArr;

    // Start is called before the first frame update
    void Start()
    {
        FormatDepthBufferID = computeShader.FindKernel("FormatDepthBuffer");
        DepthKernelID = computeShader.FindKernel("Depth");
        DrawDepthKernelID = computeShader.FindKernel("DrawDepth");
        SmoothKernelID = computeShader.FindKernel("Smooth");
        ComputeNormalsID = computeShader.FindKernel("ComputeNormals");
        TSDFUpdateID = computeShader.FindKernel("TSDFUpdate");
        RenderTSDFID = computeShader.FindKernel("RenderTSDF");
        ICPKernelID = computeShader.FindKernel("ICP");
        ICPReductionKernelID = computeShader.FindKernel("ICPReduction");
        Physics.autoSimulation = false;
        kinectVideo = new Playback("C:/Users/zhang/OneDrive/Desktop/test.mkv");
        kinectVideo.GetCalibration(out kinectCalibration);
        kinectTransform = new Transformation(kinectCalibration);
        imageWidth = kinectCalibration.DepthCameraCalibration.ResolutionWidth;
        imageHeight = kinectCalibration.DepthCameraCalibration.ResolutionHeight;

        outputImg = new Image(ImageFormat.Custom, imageWidth, imageHeight, imageWidth * 6);

        //kinectVideo.TrySeekTimestamp(K4AdotNet.Microseconds64.FromSeconds(5.8), PlaybackSeekOrigin.Begin);
        kinectVideo.TryGetNextCapture(out var capture);
        rendererComponent = GetComponent<Renderer>();
        rt = new RenderTexture(imageWidth, imageHeight, 0);
        rt.enableRandomWrite = true;
        outputTexture = new RenderTexture(imageWidth, imageHeight, 0);
        outputTexture.enableRandomWrite = true;
        RenderTexture.active = outputTexture;
        depthBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
        leftDepthBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
        smoothDepthBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
        normalMapBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
        vertexMapBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
        ICPBuffer = new ComputeBuffer(imageWidth * imageHeight * 27 / 64, 4);
        ICPReductionBuffer = new ComputeBuffer(imageWidth * imageHeight * 27 / waveGroupSize / 2, 4);
        pointCloudBuffer = new ComputeBuffer(imageWidth * imageHeight * 3 / 2, 4);
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
        pointCloudArr = new short[imageWidth * imageHeight * 3];
        imgBuffer = new byte[imageWidth * imageHeight * 4];
        smoothDepthOne = new float[imageHeight, imageWidth];
        smoothDepthTwo = new short[imageHeight / 2, imageWidth / 2];
        smoothDepthThree = new short[imageHeight / 4, imageWidth / 4];

        normalBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
        vertexBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
        tsdfBuffer = new ComputeBuffer(voxelSize * voxelSize * voxelSize, 8);
        normBufferArr = new Vector3[imageHeight, imageWidth];
        vertexBufferArr = new Vector3[imageHeight, imageWidth];
        computeShader.SetInt(imageHeightID, imageHeight);
        computeShader.SetInt(imageWidthID, imageWidth);
        computeShader.SetInt(voxelSizeID, voxelSize);
        computeShader.SetBuffer(FormatDepthBufferID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(FormatDepthBufferID, pointCloudBufferID, pointCloudBuffer);
        computeShader.SetBuffer(DepthKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(DepthKernelID, leftDepthBufferID, leftDepthBuffer);
        computeShader.SetBuffer(DrawDepthKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(DrawDepthKernelID, leftDepthBufferID, leftDepthBuffer);
        computeShader.SetTexture(DrawDepthKernelID, pixelBufferID, rt);
        computeShader.SetTexture(DrawDepthKernelID, outputBufferID, outputTexture);
        computeShader.SetBuffer(SmoothKernelID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(SmoothKernelID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(SmoothKernelID, smoothDepthBufferID, smoothDepthBuffer);
        computeShader.SetBuffer(ComputeNormalsID, normalBufferID, normalBuffer);
        computeShader.SetBuffer(ComputeNormalsID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(TSDFUpdateID, tsdfBufferID, tsdfBuffer);
        computeShader.SetBuffer(TSDFUpdateID, depthBufferID, depthBuffer);
        computeShader.SetBuffer(TSDFUpdateID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(RenderTSDFID, tsdfBufferID, tsdfBuffer);
        computeShader.SetBuffer(RenderTSDFID, normalMapBufferID, normalMapBuffer);
        computeShader.SetBuffer(RenderTSDFID, vertexMapBufferID, vertexMapBuffer);
        computeShader.SetTexture(RenderTSDFID, outputBufferID, outputTexture);
        computeShader.SetBuffer(ICPKernelID, normalBufferID, normalBuffer);
        computeShader.SetBuffer(ICPKernelID, vertexBufferID, vertexBuffer);
        computeShader.SetBuffer(ICPKernelID, normalMapBufferID, normalMapBuffer);
        computeShader.SetBuffer(ICPKernelID, vertexMapBufferID, vertexMapBuffer);
        computeShader.SetBuffer(ICPKernelID, ICPBufferID, ICPBuffer);
        computeShader.SetBuffer(ICPReductionKernelID, ICPBufferID, ICPBuffer);
        computeShader.SetBuffer(ICPReductionKernelID, ICPReductionBufferID, ICPReductionBuffer);
        tJDecompressor = new TJDecompressor();
        defaultDepthArr = new int[imageHeight * imageWidth];
        System.Array.Fill(defaultDepthArr, 1 << 20);
        Application.targetFrameRate = 60;
        //tsdfArr = new TSDF[voxelSize, voxelSize, voxelSize];
        /*
        tsdfArr = new TSDF[voxelSize, voxelSize, voxelSize];
        for (int a = 0; a < voxelSize; a++)
        {
            for (int b = 0; b < voxelSize; b++)
            {
                for (int c = 0; c < voxelSize; c++)
                {
                    tsdfArr[a, b, c] = new TSDF(1, 0);
                    //tsdfArr[a, b, c] = new TSDF((a - 170.0f) * (a - 170.0f) + (b - 160.0f) * (b - 160.0f) + (c - 128.0f) * (c - 128.0f) - 1600.0f, 0);
                }
            }
        }
        */
        //TODO: figure out why it leaves the tsdfBuffer array in gpu memory causing it to use 2x as much gpu memory and cause a gpu memory leak
        //tsdfBuffer.SetData(tsdfArr);
        //cameraMatrix = new Matrix4x4(new Vector4(1, 0, 0, 0), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(voxelSize / 2, voxelSize / 2, voxelSize / 2, 1));
        cameraMatrix = new Matrix4x4(new Vector4(1, 0, 0, 0), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(60, 60, 60, 1));
        Debug.Log(kinectCalibration.ColorCameraCalibration.Intrinsics.Parameters.Fx + " " + kinectCalibration.ColorCameraCalibration.Intrinsics.Parameters.Fy + " " + kinectCalibration.ColorCameraCalibration.Intrinsics.Parameters.Cx + " " + kinectCalibration.ColorCameraCalibration.Intrinsics.Parameters.Cy);
        Debug.Log(kinectCalibration.DepthCameraCalibration.Intrinsics.Parameters.Fx + " " + kinectCalibration.DepthCameraCalibration.Intrinsics.Parameters.Fy + " " + kinectCalibration.DepthCameraCalibration.Intrinsics.Parameters.Cx + " " + kinectCalibration.DepthCameraCalibration.Intrinsics.Parameters.Cy);
        CalibrationIntrinsicParameters param = kinectCalibration.DepthCameraCalibration.Intrinsics.Parameters;
        colorIntrinsicMatrix = new Matrix4x4(new Vector4(param.Fx, 0, 0, 0), new Vector4(0, param.Fy, 0, 0), new Vector4(param.Cx, param.Cy, 1, 0), new Vector4(0, 0, 0, 1));
        //colorIntrinsicMatrix = new Matrix4x4(new Vector4(640, 0, 0, 0), new Vector4(0, 640, 0, 0), new Vector4(640, 360, 1, 0), new Vector4(0, 0, 0, 1));
        Debug.Log(colorIntrinsicMatrix);
        Debug.Log("inverse: " + colorIntrinsicMatrix.inverse);

        normalMapArr = new Vector3[imageHeight, imageWidth];
        vertexMapArr = new Vector3[imageHeight, imageWidth];
        isTracking = false;
        isFirst = false;
        leftMatArr = new float[imageHeight * imageWidth, 6];
        rightValArr = new float[imageHeight * imageWidth];

        ICPReductionBufferArr = new float[imageHeight * imageWidth * 27 / waveGroupSize / 2];
        ICPReductionTotArr = new float[27];
        ICPReductionResultArr = new float[42];
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
        vertexBuffer.Release();
        tsdfBuffer.Release();
        ICPBuffer.Release();
        ICPReductionBuffer.Release();
        pointCloudBuffer.Release();
        outputTexture.Release();
        rt.Release();
        smoothDepthBuffer.Release();
        normalMapBuffer.Release();
        vertexMapBuffer.Release();
        kinectTransform.Dispose();
        kinectVideo.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        frame++;
        //if (frame % 2 == 0) return;
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
            //kinectTransform.DepthImageToPointCloud(depthImg, CalibrationGeometry.Depth, outputImg);
            if (img != null && depthImg != null)
            {
                /*
                unsafe
                {
                    fixed (byte* ptr = imgBuffer)
                    {
                        System.IntPtr imgBufferPtr = new System.IntPtr(ptr);
                        //Run heavy CPU computations concurrently
                        var tasks = new[]
                        {
                            Task.Run(() => DepthToPointCloudTransformationProfiler(depthImg)),
                            Task.Run(() => JPGDecompressProfiler(img, imgBufferPtr))
                        };
                        Task.WaitAll(tasks);
                    }
                }
                */
                kinectTransform.DepthImageToPointCloud(depthImg, CalibrationGeometry.Depth, outputImg);
                outputImg.CopyTo(pointCloudArr);
                pointCloudBuffer.SetData(pointCloudArr);
                computeShader.Dispatch(FormatDepthBufferID, imageWidth * imageHeight / 64, 1, 1);
                /*
                kinectTransform.DepthImageToPointCloud(depthImg, CalibrationGeometry.Depth, outputImg);
                outputImg.CopyTo(pointCloudArr);
                pointCloudBuffer.SetData(pointCloudArr);
                computeShader.Dispatch(FormatDepthBufferID, imageHeight * imageWidth / 64, 1, 1);
                */
                if (renderMode == 0)
                {
                    //PlaneRender();
                }
                else if (renderMode == 1)
                {
                    KinectFusion();
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
        //outputImg.CopyTo(colorDepth);
        //depthBuffer.SetData(colorDepth);

        leftDepthBuffer.SetData(defaultDepthArr);
        computeShader.Dispatch(DepthKernelID, imageWidth / 8, imageHeight / 8, 1);
        Graphics.Blit(tex, rt);
        Graphics.Blit(blankBackground, outputTexture);
        //draw pixels on screen
        computeShader.Dispatch(DrawDepthKernelID, imageWidth / 8, imageHeight / 8, 1);
        rendererComponent.material.mainTexture = rt;
    }

    void KinectFusion()
    {
        
        //outputImg.CopyTo(colorDepth);
        //depthBuffer.SetData(colorDepth);
        //TODO: Implement depth/normal map pyramid
        //Use bilateral filtering on depth
        //cameraMatrix = new Matrix4x4(new Vector4(1, 0, 0, leftEyeTranslationDistance), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(0, 0, 0, 1));
        computeShader.SetFloat(spatialWeightID, spatialWeight);
        computeShader.SetFloat(rangeWeightID, rangeWeight);
        computeShader.SetFloat(truncationDistID, truncationDist);
        computeShader.SetFloat(roomSizeID, roomSize);
        computeShader.SetFloat(cameraScaleID, cameraScale);
        computeShader.SetInt(neighborSizeID, neighborhoodSize);
        computeShader.SetInt(rayTraceStepsID, rayTraceSteps);
        computeShader.SetMatrix(cameraMatrixID, cameraMatrix);
        computeShader.SetMatrix(invCameraMatrixID, cameraMatrix.inverse);
        computeShader.SetMatrix(colorIntrinsicMatrixID, colorIntrinsicMatrix);
        computeShader.SetMatrix(invColorIntrinsicMatrixID, colorIntrinsicMatrix.inverse);
        computeShader.Dispatch(SmoothKernelID, imageWidth / 8, imageHeight / 8, 1);
        //calculate normals at each point
        computeShader.Dispatch(ComputeNormalsID, imageWidth / 8, imageHeight / 8, 1);

        //ICP
        
        if (isTracking)
        {
            Matrix4x4 currentCameraMatrix = cameraMatrix;
            for (int i = 0; i < 10; i++)
            {
                computeShader.SetFloat(ICPThresholdDistanceID, thresholdDistance);
                computeShader.SetFloat(ICPThresholdRotationID, Mathf.Cos(thresholdRotation));
                computeShader.SetMatrix(currentICPCameraMatrixID, currentCameraMatrix);
                computeShader.Dispatch(ICPKernelID, imageWidth / 8, imageHeight / 8, 1);
                computeShader.Dispatch(ICPReductionKernelID, imageHeight * imageWidth / waveGroupSize / 2, 1, 1);
                
                ICPReductionBuffer.GetData(ICPReductionBufferArr);
                System.Array.Fill(ICPReductionTotArr, 0);
                for (int a = 0; a < imageHeight * imageWidth / waveGroupSize / 2; a++)
                {
                    for (int b = 0; b < 27; b++)
                    {
                        ICPReductionTotArr[b] += ICPReductionBufferArr[a * 27 + b];
                    }
                }
                /*
                string output = "";
                for (int a = 0; a < 27; a++)
                    output += ICPReductionTotArr[a] + " ";
                Debug.Log(output);
                */
                for (int a = 0; a < 6; a++)
                {
                    ICPReductionResultArr[36 + a] = ICPReductionTotArr[21 + a];
                    for (int b = a; b < 6; b++)
                    {
                        ICPReductionResultArr[a * 6 + b] = ICPReductionTotArr[a * 6 - a * (a - 1) / 2 + b - a];
                        ICPReductionResultArr[b * 6 + a] = ICPReductionTotArr[a * 6 - a * (a - 1) / 2 + b - a];
                    }
                }
                
                unsafe
                {
                    fixed (float* leftMatPtr = ICPReductionResultArr)
                    {
                        fixed (float* rightValPtr = &ICPReductionResultArr[6 * 6])
                        {
                            Mat leftMat = new Mat(6, 6, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(leftMatPtr), 6 * 4);
                            Mat rightValMat = new Mat(6, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(rightValPtr), 4);
                            Mat result = new Mat(6, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                            CvInvoke.Solve(leftMat, rightValMat, result, Emgu.CV.CvEnum.DecompMethod.Cholesky);

                            float[] tempArr = new float[6];
                            result.CopyTo(tempArr);
                            for (int j = 0; j < 6; j++)
                            {
                                tempArr[j] *= -1;
                            }
                            Matrix4x4 incMat = new Matrix4x4(new Vector4(1, tempArr[2], -tempArr[1], 0),
                                                             new Vector4(-tempArr[2], 1, tempArr[0], 0),
                                                             new Vector4(tempArr[1], -tempArr[0], 1, 0),
                                                             new Vector4(tempArr[3], tempArr[4], tempArr[5], 1));
                            Debug.Log("incremental: " + incMat);
                            currentCameraMatrix = incMat * currentCameraMatrix;
                            Debug.Log("currentCameraMat: " + currentCameraMatrix);
                        }
                    }
                }
                
            }
            Debug.Log("ICP Realign Matrix: " + currentCameraMatrix * cameraMatrix.inverse);
            cameraMatrix = currentCameraMatrix;
        }
        
        //visualize normals and positions
        /*
        for (int i = 0; i < imageHeight; i++)
        {
            for (int j = 0; j < imageWidth; j++)
            {
                if (j > split)
                {
                    if (float.IsFinite(normBufferArr[imageHeight - i - 1, j].x))
                    {
                        imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)((normBufferArr[imageHeight - i - 1, j].x + 1) * 256 / 2.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)((normBufferArr[imageHeight - i - 1, j].y + 1) * 256 / 2.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)((normBufferArr[imageHeight - i - 1, j].z + 1) * 256 / 2.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
                    }
                }
                else
                {
                    if (float.IsFinite(normalMapArr[imageHeight - i - 1, j].x) && normalMapArr[imageHeight - i - 1, j].magnitude > .1)
                    {
                        imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)((normalMapArr[imageHeight - i - 1, j].x + 1) * 256 / 2.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)((normalMapArr[imageHeight - i - 1, j].y + 1) * 256 / 2.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)((normalMapArr[imageHeight - i - 1, j].z + 1) * 256 / 2.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
                    }
                }
                if (j > split)
                {
                    if (float.IsFinite(vertexBufferArr[imageHeight - i - 1, j].x) && vertexBufferArr[imageHeight - i - 1, j].magnitude > .1)
                    {
                        imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)((1 - (vertexBufferArr[imageHeight - i - 1, j].x + 128) / 400.0f) * 256);
                        imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)((1 - (vertexBufferArr[imageHeight - i - 1, j].y + 128) / 400.0f) * 256);
                        imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)((1 - (vertexBufferArr[imageHeight - i - 1, j].z + 128) / 400.0f) * 256);
                        imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
                    }
                }
                else
                {
                    if (float.IsFinite(vertexMapArr[imageHeight - i - 1, j].x) && vertexMapArr[imageHeight - i - 1, j].magnitude > .1)
                    {
                        imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)((1 - (vertexMapArr[imageHeight - i - 1, j].x) / 400.0f) * 256);
                        imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)((1 - (vertexMapArr[imageHeight - i - 1, j].y) / 400.0f) * 256);
                        imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)((1 - (vertexMapArr[imageHeight - i - 1, j].z) / 400.0f) * 256);
                        imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
                    }
                }
            }
        }
        PlaneRender();
        */
        

        //visualize difference between vertices and normals
        /*
        for (int i = 0; i < imageHeight; i++)
        {
            for (int j = 0; j < imageWidth; j++)
            {
                
                if (j > split)
                {
                    if (float.IsFinite(normBufferArr[imageHeight - i - 1, j].x) && float.IsFinite(normalMapArr[imageHeight - i - 1, j].x) && normalMapArr[imageHeight - i - 1, j].magnitude > .1)
                    {
                        imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)(255 - Mathf.Abs(normBufferArr[imageHeight - i - 1, j].x - normalMapArr[imageHeight - i - 1, j].x) * 128.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)(255 - Mathf.Abs(normBufferArr[imageHeight - i - 1, j].y - normalMapArr[imageHeight - i - 1, j].y) * 128.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)(255 - Mathf.Abs(normBufferArr[imageHeight - i - 1, j].z - normalMapArr[imageHeight - i - 1, j].z) * 128.0f);
                        imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
                    }
                }
                else
                {
                    if (float.IsFinite(vertexBufferArr[imageHeight - i - 1, j].x) && float.IsFinite(vertexMapArr[imageHeight - i - 1, j].x))
                    {
                        Vector4 cameraCenter = cameraMatrix.GetColumn(3);
                        float diffX = Mathf.Abs(vertexBufferArr[imageHeight - i - 1, j].x - vertexMapArr[imageHeight - i - 1, j].x + cameraCenter.x);
                        float diffY = Mathf.Abs(vertexBufferArr[imageHeight - i - 1, j].y - vertexMapArr[imageHeight - i - 1, j].y + cameraCenter.y);
                        float diffZ = Mathf.Abs(smoothDepthOne[imageHeight - i - 1, j] - vertexMapArr[imageHeight - i - 1, j].z + cameraCenter.z);
                        imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)(diffX > pixelThreshold ? 255 : 0);
                        imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)(diffY > pixelThreshold ? 255 : 0);
                        imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)(diffZ > pixelThreshold ? 255 : 0);
                        imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
                    }
                }
                
            }
        }
        PlaneRender();
        */

        //calculate TSDF
        computeShader.Dispatch(TSDFUpdateID, voxelSize / 8, voxelSize / 8, voxelSize / 8);
        //render TSDF
        computeShader.Dispatch(RenderTSDFID, imageWidth / 8, imageHeight / 8, 1);
        rendererComponent.material.mainTexture = outputTexture;
        isTracking = true;

        //visualize bilateral filtered normals
        /*
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
        PlaneRender();
        */
        //visualize raycast interpolated normals
        /*
        for (int i = 0; i < imageHeight; i++)
        {
            for (int j = 0; j < imageWidth; j++)
            {
                imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)((normalMapArr[imageHeight - i - 1, j].x + 256) / 2.0f * 255);
                imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)((normalMapArr[imageHeight - i - 1, j].y + 256) / 2.0f * 255);
                imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)((normalMapArr[imageHeight - i - 1, j].z + 256) / 2.0f * 255);
                imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
            }
        }
        PlaneRender();
        */
    }

    void DepthToPointCloudTransformationProfiler(Image depthImg)
    {
        Profiler.BeginThreadProfiling("updateThreads", "DepthToPointCloudTransformationProfiler Task " + Task.CurrentId);
        kinectTransform.DepthImageToPointCloud(depthImg, CalibrationGeometry.Depth, outputImg);
        outputImg.CopyTo(pointCloudArr);
        pointCloudBuffer.SetData(pointCloudArr);
        computeShader.Dispatch(FormatDepthBufferID, imageHeight * imageWidth / 64, 1, 1);
        Profiler.EndThreadProfiling();
    }

    void JPGDecompressProfiler(Image img, System.IntPtr imgBufferPtr)
    {
        Profiler.BeginThreadProfiling("updateThreads", "JPGDecompressProfiler Task " + Task.CurrentId);
        tJDecompressor.Decompress(img.Buffer, (ulong)img.SizeBytes, imgBufferPtr, img.WidthPixels * img.HeightPixels * 4, TJPixelFormats.TJPF_RGBA, TJFlags.BOTTOMUP);
        Profiler.EndThreadProfiling();
    }
}
