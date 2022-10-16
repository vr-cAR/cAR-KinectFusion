using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using K4AdotNet.Record;
using K4AdotNet.Sensor;

public class TestScript : MonoBehaviour
{
    Playback kinectVideo;
    Transformation kinectTransform;
    Calibration kinectCalibration;
    int frameCount;
    Renderer rendererComponent;
    // Start is called before the first frame update
    void Start()
    {
        kinectVideo = new Playback("C:/Users/zhang/OneDrive/Desktop/output.mkv");
        kinectVideo.GetCalibration(out kinectCalibration);
        kinectTransform = new Transformation(kinectCalibration);

        //kinectVideo.TrySeekTimestamp(K4AdotNet.Microseconds64.FromSeconds(5.8), PlaybackSeekOrigin.Begin);
        kinectVideo.TryGetNextCapture(out var capture);
        //kinectVideo.TryGetNextCapture(out capture);
        /*
        Image img = capture.ColorImage;
        Image depthImg = capture.DepthImage;
        Image outputImg = new Image(ImageFormat.Depth16, img.WidthPixels, img.HeightPixels, ImageFormats.StrideBytes(ImageFormat.Depth16, img.WidthPixels));
        kinectTransform.DepthImageToColorCamera(depthImg, outputImg);
        Image pointCloud = new Image(ImageFormat.Custom, outputImg.WidthPixels, outputImg.HeightPixels, outputImg.WidthPixels * 6);
        kinectTransform.DepthImageToPointCloud(outputImg, CalibrationGeometry.Color, pointCloud);
        short[] buffer = new short[pointCloud.SizeBytes / 2];
        pointCloud.CopyTo(buffer);
        byte[] imgBuffer = new byte[img.SizeBytes];
        img.CopyTo(imgBuffer);
        Texture2D tex = new Texture2D(2, 2);
        ImageConversion.LoadImage(tex, imgBuffer);
        Texture2D shiftedImg = new Texture2D(1280, 720);
        for (int i = 0; i < img.HeightPixels; i++)
        {
            for (int j = 0; j < img.WidthPixels; j++)
            {
                //int currentDepth = ((int)depthBuffer[i * outputImg.StrideBytes + j * 2 + 1] << 8) + depthBuffer[i * outputImg.StrideBytes + j * 2];
                int currentDepth = buffer[(i * img.WidthPixels + j) * 3 + 2];
                int worldXPos = buffer[(i * img.WidthPixels + j) * 3];
                //if (currentDepth == 0)
                //    currentDepth = 12000;
                worldXPos -= 63;
                //int worldXPos = (j - img.WidthPixels / 2) * currentDepth;
                //worldXPos -= 63 * img.WidthPixels;
                
                if (currentDepth == 0)
                {
                    shiftedImg.SetPixel(j, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                }
                else
                {
                    int newXPos = worldXPos / currentDepth + img.WidthPixels / 2;
                    shiftedImg.SetPixel(newXPos, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                }
                

                //int newXPos = worldXPos / currentDepth + img.WidthPixels / 2;
                if (currentDepth != 0)
                {
                    double newXPos = ((double)worldXPos / currentDepth * img.WidthPixels / 2 + img.WidthPixels / 2);
                    if (newXPos >= 0 && newXPos < img.WidthPixels)
                    {
                        //shiftedImg.SetPixel((int)newXPos, i, shiftedImg.GetPixel((int)newXPos, i) + Color.Lerp(Color.white, tex.GetPixel(j, img.HeightPixels - i - 1), ((float)newXPos - (int)newXPos)));
                        //shiftedImg.SetPixel((int)newXPos + 1, i, shiftedImg.GetPixel((int)newXPos + 1, i) + Color.Lerp(Color.white, tex.GetPixel(j, img.HeightPixels - i - 1), 1 - ((float)newXPos - (int)newXPos)));
                        shiftedImg.SetPixel((int)newXPos, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                        shiftedImg.SetPixel((int)newXPos + 1, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                        shiftedImg.SetPixel((int)newXPos + 2, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                        //                        shiftedImg.SetPixel(newXPos, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                    }
                }
                else
                {
                    shiftedImg.SetPixel(j, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                }
            }
        }
        shiftedImg.Apply();
        */
        frameCount = 0;
        rendererComponent = GetComponent<Renderer>();
    }
    // Update is called once per frame
    void Update()
    {
        if (frameCount % 1 == 0)
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
                    kinectTransform.DepthImageToColorCamera(depthImg, outputImg);
                    short[] colorDepth = new short[img.WidthPixels * img.HeightPixels];
                    outputImg.CopyTo(colorDepth);
                    byte[] imgBuffer = new byte[img.SizeBytes];
                    img.CopyTo(imgBuffer);
                    Texture2D tex = new Texture2D(2, 2);
                    ImageConversion.LoadImage(tex, imgBuffer);
                    Texture2D shiftedImg = new Texture2D(1280, 720);
                    /*
                    for (int i = 0; i < img.HeightPixels; i++)
                    {
                        for (int j = 0; j < img.WidthPixels; j++)
                        {
                            //int currentDepth = buffer[(i * img.WidthPixels + j) * 3 + 2];
                            int currentDepth = colorDepth[i * img.WidthPixels + j];
                            //int worldYPos = buffer[(i * img.WidthPixels + j) * 3 + 1];
                            //int worldXPos = buffer[(i * img.WidthPixels + j) * 3];
                            //if (currentDepth == 0)
                            //    currentDepth = 12000;
                            //worldXPos -= 63;
                            int worldXPos = (j - img.WidthPixels / 2) * currentDepth;
                            worldXPos -= 10 * img.WidthPixels;

                            //int newXPos = worldXPos / currentDepth + img.WidthPixels / 2;
                            if (currentDepth != 0)
                            {

                                //K4AdotNet.Float3 point = new K4AdotNet.Float3(new float[] { worldXPos, worldYPos, currentDepth });
                                //K4AdotNet.Float2 pixel = kinectCalibration.Convert3DTo2D(point, CalibrationGeometry.Color, CalibrationGeometry.Color).GetValueOrDefault();

                                //shiftedImg.SetPixel(j + 50, i, new Color(currentDepth / 50.0f / 256, currentDepth / 50.0f / 256, currentDepth / 50.0f / 256));

                                double newXPos = j - 31.5 * img.WidthPixels / currentDepth;
                                //double newXPos = ((double)worldXPos / currentDepth + img.WidthPixels / 2);
                                //double newXPos = ((double)worldXPos / currentDepth * img.WidthPixels / 2 + img.WidthPixels / 2);
                                //int newYPos = (int)((double)worldYPos / currentDepth * img.HeightPixels / 2 + img.HeightPixels / 2);
                                if (newXPos >= 0 && newXPos < img.WidthPixels)
                                {
                                    //shiftedImg.SetPixel(j - 50, i, Color.red);
                                    //shiftedImg.SetPixel(Mathf.RoundToInt((float)newXPos), i, tex.GetPixel(j, img.HeightPixels - i - 1));
                                    //shiftedImg.SetPixel((int)newXPos + 0, i, Color.cyan);
                                    //shiftedImg.SetPixel((int)newXPos + 1, i, Color.cyan);
                                    shiftedImg.SetPixel((int)newXPos + 0, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                                    shiftedImg.SetPixel((int)newXPos + 1, i, tex.GetPixel(j, img.HeightPixels - i - 1));

                                    //shiftedImg.SetPixel((int)newXPos, i, shiftedImg.GetPixel((int)newXPos, i) + Color.Lerp(Color.black, tex.GetPixel(j, img.HeightPixels - i - 1), ((float)newXPos - (int)newXPos)));
                                    //shiftedImg.SetPixel((int)newXPos + 1, i, shiftedImg.GetPixel((int)newXPos + 1, i) + Color.Lerp(Color.black, tex.GetPixel(j, img.HeightPixels - i - 1), 1 - ((float)newXPos - (int)newXPos)));
                                    //shiftedImg.SetPixel((int)newXPos, newYPos, tex.GetPixel(j, img.HeightPixels - i - 1));
                                    //shiftedImg.SetPixel((int)newXPos + 1, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                                    //shiftedImg.SetPixel((int)newXPos + 2, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                                    //                        shiftedImg.SetPixel(newXPos, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                                }
                                
                            }
                            else
                            {
                                //shiftedImg.SetPixel(j, i, tex.GetPixel(j, img.HeightPixels - i - 1));
                            }
                        }
                    }
                    shiftedImg.Apply();
                    rendererComoponent.material.mainTexture = shiftedImg;
                    */
                    rendererComponent.material.mainTexture = tex;
                    //GetComponent<Renderer>().material.mainTexture = shiftedImg;
                }
            }
            
            /*
            if (capture != null)
            {
                Image img = capture.ColorImage;
                Image depthImg = capture.DepthImage;
                if (img != null)
                {
                    if (depthImg != null)
                    {
                        Image outputImg = new Image(ImageFormat.Depth16, img.WidthPixels, img.HeightPixels, ImageFormats.StrideBytes(ImageFormat.Depth16, img.WidthPixels));
                        kinectTransform.DepthImageToColorCamera(depthImg, outputImg);
                        byte[] depthBuffer = new byte[outputImg.HeightPixels * outputImg.StrideBytes];
                        outputImg.CopyTo(depthBuffer);
                        Texture2D tex = new Texture2D(outputImg.WidthPixels, outputImg.HeightPixels, TextureFormat.RGB24, false, true);
                        byte[] arr = new byte[outputImg.HeightPixels * outputImg.WidthPixels * 3];
                        for (int i = outputImg.HeightPixels - 1; i >= 0; i--)
                        {
                            for (int j = 0; j < outputImg.WidthPixels; j++)
                            {
                                int color = (int)depthBuffer[i * outputImg.StrideBytes + j * 2 + 1] * 256 + (int)depthBuffer[i * outputImg.StrideBytes + j * 2];
                                color /= 40;
                                byte colorByte = (byte)color;
                                arr[i * outputImg.WidthPixels * 3 + j * 3 + 0] = colorByte;
                                arr[i * outputImg.WidthPixels * 3 + j * 3 + 1] = colorByte;
                                arr[i * outputImg.WidthPixels * 3 + j * 3 + 2] = colorByte;
                                //                            arr[i * img.WidthPixels * 3 + j * 3] = 255;
                            }
                        }
                        tex.LoadRawTextureData(arr);
                        tex.Apply();
                        GetComponent<Renderer>().material.mainTexture = tex;
                    }
                    else
                    {
                        Texture2D tex = new Texture2D(1280, 720, TextureFormat.RGBA32, false);
                        byte[] arr = new byte[img.SizeBytes];
                        Debug.Log(img.CopyTo(arr) + " " + img.SizeBytes + " " + img.WidthPixels + " " + img.HeightPixels);
                        tex.LoadImage(arr);
                        Debug.Log(tex.height + " " + tex.width);
                        GetComponent<Renderer>().material.mainTexture = tex;
                    }
                }
            }
            */
            /*
            if (capture != null)
            {
                Image img = capture.DepthImage;
                if (img != null)
                {
                    byte[] depthBuffer = new byte[img.HeightPixels * img.StrideBytes];
                    img.CopyTo(depthBuffer);
                    Texture2D tex = new Texture2D(img.WidthPixels, img.HeightPixels, TextureFormat.RGB24, false, true);
                    byte[] arr = new byte[img.HeightPixels * img.WidthPixels * 3];
                    for (int i = img.HeightPixels - 1; i >= 0; i--)
                    {
                        for (int j = 0; j < img.WidthPixels; j++)
                        {
                            int color = (int) depthBuffer[i * img.StrideBytes + j * 2 + 1] * 256 + (int) depthBuffer[i * img.StrideBytes + j * 2];
                            color /= 40;
                            byte colorByte = (byte)color;
                            arr[i * img.WidthPixels * 3 + j * 3 + 0] = colorByte;
                            arr[i * img.WidthPixels * 3 + j * 3 + 1] = colorByte;
                            arr[i * img.WidthPixels * 3 + j * 3 + 2] = colorByte;
                            //                            arr[i * img.WidthPixels * 3 + j * 3] = 255;
                        }
                    }
                    tex.LoadRawTextureData(arr);
                    tex.Apply();
//                    Debug.Log(tex.LoadImage(arr));
                    GetComponent<Renderer>().material.mainTexture = tex;
                }
            }
            */
        }
        frameCount++;
    }
}
