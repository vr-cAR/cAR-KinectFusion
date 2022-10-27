using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Emgu.CV;

public class NewBehaviourScript : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        int size = 10000;
        
        float[,] leftMat = new float[6, 6];
        float[] rightVal = new float[6];
        
        /*
        float[,] leftMat = new float[6 * size, 6];
        float[] rightVal = new float[6 * size];
        */
        for (int i = 0; i < size; i++)
        {
            //Vector3 prevNormal = new Vector3(1 + Random.Range(0f, .0001f), Random.Range(0f, .0001f), Random.Range(0f, .0001f)).normalized;
            //Vector3 prevNormal = new Vector3(1 + Random.Range(0f, .01f), Random.Range(0f, .01f), Random.Range(0f, .01f)).normalized;
            Vector3 prevNormal = new Vector3(1, 0, 0).normalized;
            Vector3 prevVertex = new Vector3(Random.Range(-100, 100), Random.Range(-100, 100), Random.Range(-100, 100));
            Vector3 estimateVertex = prevVertex + prevNormal;
            float[] ATransposeMatrix = new float[] { prevNormal.y * estimateVertex.z - prevNormal.z * estimateVertex.y,
                                                             estimateVertex.x * prevNormal.z - estimateVertex.z * prevNormal.x,
                                                             estimateVertex.y * prevNormal.x - estimateVertex.x * prevNormal.y,
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
        }

        unsafe
        {
            fixed (float* ptrTwo = rightVal)
            {
                fixed (float* ptr = leftMat)
                {
                    /*
                    string printOut = "";
                    for (int a = 0; a < 6 * size; a++)
                    {
                        for (int b = 0; b < 6; b++)
                        {
                            printOut += leftMat[a, b] + " ";
                        }
                        printOut += "\n";
                    }
                    Debug.Log(printOut);
                    printOut = "";
                    for (int a = 0; a < 6 * size; a++)
                    {
                        printOut += rightVal[a] + " ";
                    }
                    Debug.Log(printOut);
                    */
                    Mat leftArr = new Mat(6, 6, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptr), 6 * 4);
                    Mat rightValArr = new Mat(6, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1, new System.IntPtr(ptrTwo), 4);
                    Mat result = new Mat(6, 1, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
                    Debug.Log(CvInvoke.Solve(leftArr, rightValArr, result, Emgu.CV.CvEnum.DecompMethod.Normal) ? "Found a solution" : "No solution found");
                    float[] tempArr = new float[6];
                    result.CopyTo(tempArr);
                    Matrix4x4 incMat = new Matrix4x4(new Vector4(1, -tempArr[2], tempArr[1], 0),
                                                     new Vector4(tempArr[2], 1, -tempArr[0], 0),
                                                     new Vector4(-tempArr[1], tempArr[0], 1, 0),
                                                     new Vector4(tempArr[3], tempArr[4], tempArr[5], 1));
                    Debug.Log("incremental: " + incMat);
                }
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
