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
        vertexMapBufferID = Shader.PropertyToID("vertexMapBuffer");
    RenderTexture rt;
    RenderTexture outputTexture;
    Texture2D tex;
    Texture2D blankBackground;
    int[] defaultDepthArr;
    short[] colorDepth;
    byte[] imgBuffer;
    Vector3[,] normBufferArr;
    Vector3[,] vertexBufferArr;
    float[,] smoothDepthOne;
    short[,] smoothDepthTwo;
    short[,] smoothDepthThree;
    TJDecompressor tJDecompressor;
    Image outputImg;
    int DepthKernelID;
    int DrawDepthKernelID;
    int SmoothKernelID;
    int ComputeNormalsID;
    int TSDFUpdateID;
    int RenderTSDFID;
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
    Matrix4x4 prevCameraMatrix;
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

    // Start is called before the first frame update
    void Start()
    {
        DepthKernelID = computeShader.FindKernel("Depth");
        DrawDepthKernelID = computeShader.FindKernel("DrawDepth");
        SmoothKernelID = computeShader.FindKernel("Smooth");
        ComputeNormalsID = computeShader.FindKernel("ComputeNormals");
        TSDFUpdateID = computeShader.FindKernel("TSDFUpdate");
        RenderTSDFID = computeShader.FindKernel("RenderTSDF");
        Physics.autoSimulation = false;
        kinectVideo = new Playback("C:/Users/zhang/OneDrive/Desktop/test.mkv");
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
        smoothDepthBuffer = new ComputeBuffer(imageWidth * imageHeight, 4);
        normalMapBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
        vertexMapBuffer = new ComputeBuffer(imageWidth * imageHeight, 12);
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
        CalibrationIntrinsicParameters param = kinectCalibration.ColorCameraCalibration.Intrinsics.Parameters;
        //colorIntrinsicMatrix = new Matrix4x4(new Vector4(param.Fx, 0, 0, 0), new Vector4(0, param.Fy, 0, 0), new Vector4(param.Cx, param.Cy, 1, 0), new Vector4(0, 0, 0, 1));
        colorIntrinsicMatrix = new Matrix4x4(new Vector4(640, 0, 0, 0), new Vector4(0, 640, 0, 0), new Vector4(640, 360, 1, 0), new Vector4(0, 0, 0, 1));
        Debug.Log(colorIntrinsicMatrix);
        Debug.Log("inverse: " + colorIntrinsicMatrix.inverse);

        normalMapArr = new Vector3[imageHeight, imageWidth];
        vertexMapArr = new Vector3[imageHeight, imageWidth];
        isTracking = false;
        isFirst = false;
        leftMatArr = new float[imageHeight * imageWidth, 6];
        rightValArr = new float[imageHeight * imageWidth];
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
        if (frame % 2 == 0) return;
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
            
            float[,] leftMat = new float[6, 6];
            float[] rightVal = new float[6];
            //BufferArr is new data, MapArr is data from previous frame from ray casting
            
            vertexBuffer.GetData(vertexBufferArr);
            normalBuffer.GetData(normBufferArr);
            smoothDepthBuffer.GetData(smoothDepthOne);
            

            if (!isFirst)
            {
                /*
                using(StreamWriter writeOne = new StreamWriter("C:/Users/zhang/OneDrive/Desktop/newPointCloudNormOne.obj"))
                {
                    using (StreamWriter writeTwo = new StreamWriter("C:/Users/zhang/OneDrive/Desktop/prevPointCloudNormOne.obj"))
                    {
                        for (int i = 0; i < imageHeight; i++)
                        {
                            for (int j = 0; j < imageWidth; j++)
                            {
                                if (vertexBufferArr[i, j].magnitude > .1)
                                {
                                    writeOne.WriteLine("v " + vertexBufferArr[i, j].x + " " + vertexBufferArr[i, j].y + " " + vertexBufferArr[i, j].z);
                                }
                                if (vertexMapArr[i, j].magnitude > .1)
                                {
                                    writeTwo.WriteLine("v " + vertexMapArr[i, j].x + " " + vertexMapArr[i, j].y + " " + vertexMapArr[i, j].z);
                                }
                            }
                        }
                        
                        writeOne.WriteLine("ply");
                        writeOne.WriteLine("format ascii 1.0");
                        writeOne.WriteLine("element vertex \\(vertexCount)");
                        writeOne.WriteLine("property float x");
                        writeOne.WriteLine("property float y");
                        writeOne.WriteLine("property float z");
                        writeOne.WriteLine("property float nx");
                        writeOne.WriteLine("property float ny");
                        writeOne.WriteLine("property float nz");
                        writeOne.WriteLine("end_header");
                        writeTwo.WriteLine("ply");
                        writeTwo.WriteLine("format ascii 1.0");
                        writeTwo.WriteLine("element vertex \\(vertexCount)");
                        writeTwo.WriteLine("property float x");
                        writeTwo.WriteLine("property float y");
                        writeTwo.WriteLine("property float z");
                        writeTwo.WriteLine("property float nx");
                        writeTwo.WriteLine("property float ny");
                        writeTwo.WriteLine("property float nz");
                        writeTwo.WriteLine("end_header");
                        for (int i = 0; i < imageHeight; i++)
                        {
                            for (int j = 0; j < imageWidth; j++)
                            {
                                if (!(vertexBufferArr[i, j].z == 0 || vertexBufferArr[i, j].z == float.NaN) && float.IsFinite(normBufferArr[i, j].x))
                                {
                                    //writeOne.WriteLine(vertexBufferArr[i, j].x + " " + vertexBufferArr[i, j].y + " " + vertexBufferArr[i, j].z + " " + Mathf.Abs(Mathf.RoundToInt(normBufferArr[i, j].x * 255)) + " " + Mathf.Abs(Mathf.RoundToInt(normBufferArr[i, j].y * 255)) + " " + Mathf.Abs(Mathf.RoundToInt(normBufferArr[i, j].z * 255)));
                                    writeOne.WriteLine(vertexBufferArr[i, j].x + " " + vertexBufferArr[i, j].y + " " + vertexBufferArr[i, j].z + " " + normBufferArr[i, j].x + " " + normBufferArr[i, j].y + " " + normBufferArr[i, j].z);
                                }
                                if (vertexMapArr[i, j].x != 0)
                                {
                                    //writeTwo.WriteLine(vertexMapArr[i, j].x + " " + vertexMapArr[i, j].y + " " + vertexMapArr[i, j].z + " " + Mathf.RoundToInt(normalMapArr[i, j].x * 255) + " " + Mathf.RoundToInt(normalMapArr[i, j].y * 255) + " " + Mathf.RoundToInt(normalMapArr[i, j].z * 255));
                                    writeTwo.WriteLine(vertexMapArr[i, j].x + " " + vertexMapArr[i, j].y + " " + vertexMapArr[i, j].z + " " + normalMapArr[i, j].x + " " + normalMapArr[i, j].y + " " + normalMapArr[i, j].z);
                                }
                            }
                        }
                        
                    }
                }
                isFirst = true;
                */
            }
            /*
            ICP test = new ICP(10);
            float[,] srcArr = new float[10000, 6];
            float[,] dstArr = new float[10000, 6];
            int count = 0;
            for (int i = 0; i < imageHeight; i++)
            {
                if (count >= 10000) break;
                for (int j = 0; j < imageWidth; j++)
                {
                    if (count >= 10000) break;
                    if (vertexMapArr[i, j].magnitude < .01 || vertexBufferArr[i, j].magnitude < .01 || !float.IsFinite(normalMapArr[i, j].z) || !float.IsFinite(normBufferArr[i, j].z)) continue;
                    srcArr[count, 0] = vertexMapArr[i, j].x;
                    srcArr[count, 1] = vertexMapArr[i, j].y;
                    srcArr[count, 2] = vertexMapArr[i, j].z;
                    srcArr[count, 3] = normalMapArr[i, j].x;
                    srcArr[count, 4] = normalMapArr[i, j].y;
                    srcArr[count, 5] = normalMapArr[i, j].z;

                    dstArr[count, 0] = vertexBufferArr[i, j].x + 128;
                    dstArr[count, 1] = vertexBufferArr[i, j].y + 128;
                    dstArr[count, 2] = vertexBufferArr[i, j].z + 128;
                    dstArr[count, 3] = normBufferArr[i, j].x;
                    dstArr[count, 4] = normBufferArr[i, j].y;
                    dstArr[count, 5] = normBufferArr[i, j].z;
                    count++;
                }
            }
            unsafe
            {
                fixed(float* ptrOne = srcArr)
                {
                    fixed(float* ptrTwo = dstArr)
                    {
                        Mat srcMat = new Mat(10000, 6, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptrOne), 6 * 4);
                        Mat dstMat = new Mat(10000, 6, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptrTwo), 6 * 4);
                        Mat pose = new Mat(4, 4, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                        double error = -1;
                        Debug.Log(test.RegisterModelToScene(srcMat, dstMat, ref error, pose));
                        Debug.Log("error: " + error);
                        float[] arr = new float[16];
                        pose.CopyTo(arr);
                        string output = "";
                        for (int i = 0; i < 16; i++)
                        {
                            output += arr[i] + " ";
                        }
                        Debug.Log(output);
                    }
                }
            }
            */
            
            Matrix4x4 currentCameraMatrix = cameraMatrix;
            int count = 0;
            int countOne = 0;
            int countTwo = 0;
            int countThree = 0;
            for (int k = 0; k < 10; k++)
            {
                int size = 0;
                float distErrorX = 0;
                float distErrorY = 0;
                float distErrorZ = 0;
                for (int i = 0; i < imageHeight; i++)
                {
                    for (int j = 0; j < imageWidth; j++)
                    {
                        Vector4 currentVertex = new Vector4(vertexBufferArr[imageHeight - i - 1, j].x, vertexBufferArr[imageHeight - i - 1, j].y, vertexBufferArr[imageHeight - i - 1, j].z, 1);
                        if (currentVertex.z == 0) continue;
                        Matrix4x4 FrameToFrameTransform = cameraMatrix.inverse * currentCameraMatrix;
                        Vector4 projPoint = colorIntrinsicMatrix * FrameToFrameTransform * currentVertex;
                        int newPointX = Mathf.RoundToInt(projPoint.x / projPoint.z);
                        int newPointY = Mathf.RoundToInt(projPoint.y / projPoint.z);
                        //int newPointX = j;
                        //int newPointY = imageHeight - i - 1;
                        if (newPointX < 0 || newPointX >= imageWidth || newPointY < 0 || newPointY >= imageHeight)
                        {
                            countOne++;
                            continue;
                        }
                        if (vertexMapArr[newPointY, newPointX].z == 0) continue;
                        Vector3 tempVertex = vertexMapArr[newPointY, newPointX];
                        Vector4 prevVertex = new Vector4(tempVertex.x, tempVertex.y, tempVertex.z, 1);

                        if (Vector4.Distance(currentCameraMatrix * currentVertex, prevVertex) > thresholdDistance)
                        {
                            countTwo++;
                            continue;
                        }
                        Vector4 currentNormal = new Vector4(normBufferArr[imageHeight - i - 1, j].x, normBufferArr[imageHeight - i - 1, j].y, normBufferArr[imageHeight - i - 1, j].z, 0);
                        Vector3 tempNormal = normalMapArr[newPointY, newPointX];
                        if (!float.IsFinite(currentNormal.z) || currentNormal.magnitude < .1 || !float.IsFinite(tempNormal.z) || tempNormal.magnitude < .1) continue;
                        Vector4 prevNormal = new Vector4(tempNormal.x, tempNormal.y, tempNormal.z, 0);
                        if (Mathf.Abs(Vector4.Dot(currentCameraMatrix * currentNormal, prevNormal)) < Mathf.Cos(thresholdRotation))
                        {
                            countThree++;
                            continue;
                        }
                        currentNormal = currentCameraMatrix * currentNormal;
                        count++;
                        Vector4 estimateVertex = currentCameraMatrix * currentVertex;
                        distErrorX += Mathf.Abs(estimateVertex.x - prevVertex.x);
                        distErrorY += Mathf.Abs(estimateVertex.y - prevVertex.y);
                        distErrorZ += Mathf.Abs(estimateVertex.z - prevVertex.z);
                        float[] ATransposeMatrix = new float[] { estimateVertex.y * currentNormal.z - estimateVertex.z * currentNormal.y,
                                                             estimateVertex.z * currentNormal.x - estimateVertex.x * currentNormal.z,
                                                             estimateVertex.x * currentNormal.y - estimateVertex.y * currentNormal.x,
                                                             currentNormal.x,
                                                             currentNormal.y,
                                                             currentNormal.z};
                        float bScalar = currentNormal.x * (estimateVertex.x - prevVertex.x) + currentNormal.y * (estimateVertex.y - prevVertex.y) + currentNormal.z * (estimateVertex.z - prevVertex.z);
                        for (int a = 0; a < 6; a++)
                            leftMatArr[size, a] = ATransposeMatrix[a];
                        rightValArr[size] = bScalar;
                        size++;
                        /*
                        float[] ATransposeMatrix = new float[] { estimateVertex.y * prevNormal.z - estimateVertex.z * prevNormal.y,
                                                             estimateVertex.z * prevNormal.x - estimateVertex.x * prevNormal.z,
                                                             estimateVertex.x * prevNormal.y - estimateVertex.y * prevNormal.x,
                                                             prevNormal.x,
                                                             prevNormal.y,
                                                             prevNormal.z};
                        float bScalar = prevNormal.x * (prevVertex.x - estimateVertex.x) + prevNormal.y * (prevVertex.y - estimateVertex.y) + prevNormal.z * (prevVertex.z - estimateVertex.z);
                        for (int a = 0; a < 6; a++)
                        {
                            rightVal[a] += bScalar * ATransposeMatrix[a];
                            for (int b = 0; b < 6; b++)
                            {
                                leftMat[a, b] += ATransposeMatrix[a] * ATransposeMatrix[b];
                            }
                        }
                        */

                        //Vector3 tempNorm = normalMapArr[imageHeight - i - 1, j];
                        //imgBuffer[(i * imageWidth + j) * 4 + 0] = (byte)(Mathf.Abs(tempNormal.x) * 255);
                        //imgBuffer[(i * imageWidth + j) * 4 + 1] = (byte)(Mathf.Abs(tempNormal.y) * 255);
                        //imgBuffer[(i * imageWidth + j) * 4 + 2] = (byte)(Mathf.Abs(tempNormal.z) * 255);
                        imgBuffer[(i * imageWidth + j) * 4 + 0] = 255;
                        imgBuffer[(i * imageWidth + j) * 4 + 1] = 255;
                        imgBuffer[(i * imageWidth + j) * 4 + 2] = 255;

                        imgBuffer[(i * imageWidth + j) * 4 + 3] = 255;
                    }
                }
                Debug.Log(distErrorX / size + " " + distErrorY / size + " " + distErrorZ / size);
                unsafe
                {
                    //fixed (float* ptrTwo = rightVal)
                    fixed (float* ptrTwo = rightValArr)
                    {
                        //fixed (float* ptr = leftMat)
                        fixed (float* ptr = leftMatArr)
                        {
                            Mat leftArr = new Mat(size, 6, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptr), 6 * 4);
                            Mat rightValArr = new Mat(size, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptrTwo), 4);
                            Mat result = new Mat(6, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                            CvInvoke.Solve(leftArr, rightValArr, result, Emgu.CV.CvEnum.DecompMethod.Svd);
                            /*
                            Mat leftArr = new Mat(6, 6, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptr), 6 * 4);
                            Mat rightValArr = new Mat(6, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptrTwo), 4);
                            Mat result = new Mat(6, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                            CvInvoke.Solve(leftArr, rightValArr, result, Emgu.CV.CvEnum.DecompMethod.Cholesky);
                            */
                            float[] tempArr = new float[6];
                            result.CopyTo(tempArr);
                            for (int i = 0; i < 6; i++)
                            {
                                tempArr[i] *= -1;
                            }
                            
                            Matrix4x4 incMat = new Matrix4x4(new Vector4(1, tempArr[2], -tempArr[1], 0),
                                                             new Vector4(-tempArr[2], 1, tempArr[0], 0),
                                                             new Vector4(tempArr[1], -tempArr[0], 1, 0),
                                                             new Vector4(tempArr[3], tempArr[4], tempArr[5], 1));
                            /*
                            Matrix4x4 incMat = new Matrix4x4(new Vector4(1, -tempArr[2], tempArr[1], 0),
                                                             new Vector4(tempArr[2], 1, -tempArr[0], 0),
                                                             new Vector4(-tempArr[1], tempArr[0], 1, 0),
                                                             new Vector4(-tempArr[3], -tempArr[4], -tempArr[5], 1));
                            */
                            /*
                            Matrix4x4 incMat = new Matrix4x4(new Vector4(1, tempArr[2], -tempArr[1], 0),
                                                             new Vector4(tempArr[0] * tempArr[1] - tempArr[2], tempArr[0] * tempArr[1] * tempArr[2] + 1, tempArr[0], 0),
                                                             new Vector4(tempArr[0] * tempArr[2] + tempArr[1], tempArr[1] * tempArr[2] - tempArr[0], 1, 0),
                                                             new Vector4(tempArr[3], tempArr[4], tempArr[5], 1));
                            */
                            Debug.Log("incremental: " + incMat);
                            currentCameraMatrix = incMat * currentCameraMatrix;
                            Debug.Log("currentCameraMat: " + currentCameraMatrix);
                        }
                    }
                }
            }
            //PlaneRender();
            cameraMatrix = currentCameraMatrix;
            Debug.Log(count + " " + countOne + " " + countTwo + " " + countThree);
            
            /*
            string printOut = "";
            for (int a = 0; a < 6; a++)
            {
                for (int b = 0; b < 6; b++)
                {
                    printOut += (long)leftMat[a, b] + " ";
                }
                printOut += "\n";
            }
            Debug.Log(printOut);
            printOut = "";
            for (int a = 0; a < 6; a++)
            {
                printOut += (long)rightVal[a] + " ";
            }
            Debug.Log(printOut);
            PlaneRender();
            */
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
        
        normalMapBuffer.GetData(normalMapArr);
        vertexMapBuffer.GetData(vertexMapArr);
        prevCameraMatrix = cameraMatrix;
        
        /*
        if (frame > 100 && !isFirst)
        {
            Debug.Log("Writing");
            isFirst = true;
            tsdfBuffer.GetData(tsdfArr);
            using (StreamWriter writeOne = new StreamWriter("C:/Users/zhang/OneDrive/Desktop/tsdfArr.obj"))
            {
                for (int i = 0; i < voxelSize; i++)
                {
                    for (int j = 0; j < voxelSize; j++)
                    {
                        for (int k = 0; k < voxelSize; k++)
                        {
                            if (tsdfArr[i, j, k].tsdfValue < 0)
                                writeOne.WriteLine("v " + i + " " + j + " " + k);
                        }
                    }
                }
            }
            Debug.Log("Writing end");
        }
        */
        /*
        smoothDepthBuffer.GetData(smoothDepthOne);
        for (int i = 0; i < imageHeight / 2; i ++)
        {
            for (int j = 0; j < imageWidth / 2; j ++)
            {
                float totDepth = 0;
                int count = 0;
                short origDepth = smoothDepthOne[i * 2 * imageWidth + j * 2];
                for (int a = -1; a <= 1; a++)
                {
                    for (int b = -1; b <= 1; b++)
                    {
                        int origDepthPosRow = i * 2 + a;
                        int origDepthPosCol = j * 2 + b;
                        if (origDepthPosRow < 0 || origDepthPosRow >= imageHeight || origDepthPosCol < 0 || origDepthPosCol >= imageWidth)
                            continue;
                        if (Mathf.Abs(smoothDepthOne[origDepthPosRow * imageWidth + origDepthPosCol] - origDepth) <= 3 * rangeWeight)
                        {
                            totDepth += smoothDepthOne[origDepthPosRow * imageWidth + origDepthPosCol];
                            count++;
                        }
                    }
                }
                smoothDepthTwo[i * imageWidth / 2 + j] = (short)(totDepth / count);
            }
        }
        */

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
