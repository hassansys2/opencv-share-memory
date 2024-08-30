using System;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

class Program
{
    const int width = 640;      // cols
    const int height = 480;     // rows
    const int channels = 3;
    const int matrixSize = width * height * channels;

    static void Main()
    {
        CvInvoke.NamedWindow("Checkerboard", WindowFlags.Normal);
        VideoCapture capture = new VideoCapture(0);
        capture.Set(CapProp.FrameWidth, width);
        capture.Set(CapProp.FrameHeight, height);
        Mat matrix = new Mat();

        using (MemoryMappedFile sharedMemory = MemoryMappedFile.CreateOrOpen("FrameBuffer", matrixSize))
        {
            while (true)
            {
                capture.Read(matrix);
                
                if (matrix.IsEmpty)continue;

                using (MemoryMappedViewAccessor accessor = sharedMemory.CreateViewAccessor(0, matrixSize))
                {
                    byte[] matrixBytes = new byte[matrixSize];
                    Marshal.Copy(matrix.DataPointer, matrixBytes, 0, matrixBytes.Length);
                    accessor.WriteArray(0, matrixBytes, 0, matrixBytes.Length);
                }
                
                CvInvoke.Imshow("Checkerboard", matrix);
                
                int key = CvInvoke.WaitKey(1);

                if (key == 'q' || key == 'Q')
                {
                    break;
                }
            }
        }

        capture.Dispose();
        CvInvoke.DestroyAllWindows();
    }
}
