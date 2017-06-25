//------------------------------------------------------------------------------
// <copyright file="CoordinateMappingBasics.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#include "stdafx.h"
#include <strsafe.h>
#include <math.h>
#include <limits>
#include <Wincodec.h>
#include "resource.h"
#include "CoordinateMappingBasics.h"
#include <string>


#ifndef HINST_THISCOMPONENT
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#define HINST_THISCOMPONENT ((HINSTANCE)&__ImageBase)
#endif


USHORT nDepthMinReliableDistance = 0;
USHORT nDepthMaxDistance = 0;

std::vector<cv::Mat> ColorImagesQueue;
std::vector<cv::Mat> depthImagesQueue;
std::vector<cv::Mat> maskImagesQueue;

int kinect_read_frame_counter = 0;

// std::vector<int> frame_counter_global;
/// <summary>
/// Entry point for the application
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="hPrevInstance">always 0</param>
/// <param name="lpCmdLine">command line arguments</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
/// <returns>status</returns>
/*int APIENTRY wWinMain(
_In_ HINSTANCE hInstance,
_In_opt_ HINSTANCE hPrevInstance,
_In_ LPWSTR lpCmdLine,
_In_ int nShowCmd
)
{
UNREFERENCED_PARAMETER(hPrevInstance);
UNREFERENCED_PARAMETER(lpCmdLine);

CCoordinateMappingBasics application;
application.Run(hInstance, nShowCmd);
}

*/

/// <summary>
/// Constructor
/// </summary>
CCoordinateMappingBasics::CCoordinateMappingBasics() :
m_hWnd(NULL),
m_nStartTime(0),
m_nLastCounter(0),
m_nFramesSinceUpdate(0),
m_fFreq(0),
m_nNextStatusTime(0LL),
m_bSaveScreenshot(true),
m_pKinectSensor(NULL),
m_pCoordinateMapper(NULL),
m_pMultiSourceFrameReader(NULL),
m_pDepthCoordinates(NULL),
m_pColorCoordinates(NULL),
m_pD2DFactory(NULL),
m_pDrawCoordinateMapping(NULL),
m_pOutputRGBX(NULL),
m_pBackgroundRGBX(NULL),
m_pBackgroundRGBX_depthSpace(NULL),
m_pColorRGBX(NULL),
m_pDepthRGBX(NULL)
{
	LARGE_INTEGER qpf = { 0 };
	if (QueryPerformanceFrequency(&qpf))
	{
		m_fFreq = double(qpf.QuadPart);
	}

	// create heap storage for composite image pixel data in RGBX format
	m_pOutputRGBX = new RGBQUAD[cColorWidth * cColorHeight];

	// create heap storage for composite image pixel data in RGBX format
	m_pOutputRGBX_depthSpace = new RGBQUAD[cDepthWidth * cDepthHeight];

	// create heap storage for background image pixel data in RGBX format
	m_pBackgroundRGBX = new RGBQUAD[cColorWidth * cColorHeight];

	// create heap storage for background image pixel data in RGBX format
	m_pBackgroundRGBX_depthSpace = new RGBQUAD[cDepthWidth * cDepthHeight];

	// create heap storage for color pixel data in RGBX format
	m_pColorRGBX = new RGBQUAD[cColorWidth * cColorHeight];

	// create heap storage for the coorinate mapping from color to depth
	m_pDepthCoordinates = new DepthSpacePoint[cColorWidth * cColorHeight];

	// create heap storage for the coorinate mapping from depth to color
	m_pColorCoordinates = new ColorSpacePoint[cDepthWidth * cDepthHeight];

	// create heap storage for depth pixel data in RGBX format
	m_pDepthRGBX = new RGBQUAD[cDepthWidth * cDepthHeight];
}


/// <summary>
/// Destructor
/// </summary>
CCoordinateMappingBasics::~CCoordinateMappingBasics()
{
	// clean up Direct2D renderer
	if (m_pDrawCoordinateMapping)
	{
		delete m_pDrawCoordinateMapping;
		m_pDrawCoordinateMapping = NULL;
	}

	if (m_pOutputRGBX)
	{
		delete[] m_pOutputRGBX;
		m_pOutputRGBX = NULL;
	}

	if (m_pBackgroundRGBX)
	{
		delete[] m_pBackgroundRGBX;
		m_pBackgroundRGBX = NULL;
	}
	if (m_pBackgroundRGBX_depthSpace)
	{
		delete[] m_pBackgroundRGBX_depthSpace;
		m_pBackgroundRGBX_depthSpace = NULL;
	}
	if (m_pDepthRGBX)
	{
		delete[] m_pDepthRGBX;
		m_pDepthRGBX = NULL;
	}

	if (m_pColorRGBX)
	{
		delete[] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}

	if (m_pDepthCoordinates)
	{
		delete[] m_pDepthCoordinates;
		m_pDepthCoordinates = NULL;
	}
	if (m_pColorCoordinates)
	{
		delete[] m_pColorCoordinates;
		m_pColorCoordinates = NULL;
	}

	// clean up Direct2D
	SafeRelease(m_pD2DFactory);

	// done with frame reader
	SafeRelease(m_pMultiSourceFrameReader);

	// done with coordinate mapper
	SafeRelease(m_pCoordinateMapper);

	// close the Kinect Sensor
	if (m_pKinectSensor)
	{
		m_pKinectSensor->Close();
	}

	SafeRelease(m_pKinectSensor);
}

/// <summary>
/// Creates the main window and begins processing
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
//int CCoordinateMappingBasics::Run(HINSTANCE hInstance, int nCmdShow)
int CCoordinateMappingBasics::Run()
{
	if (m_pBackgroundRGBX)
	{
		if (FAILED(LoadResourceImage(L"Background", L"Image", cColorWidth, cColorHeight, m_pBackgroundRGBX)))
		{
			const RGBQUAD c_green = { 0, 255, 0 };

			// Fill in with a background colour of green if we can't load the background image
			for (int i = 0; i < cColorWidth * cColorHeight; ++i)
			{
				m_pBackgroundRGBX[i] = c_green;
			}
		}
	}

	if (m_pBackgroundRGBX_depthSpace)
	{
		//if (FAILED(LoadResourceImage(L"Background", L"Image", cDepthWidth, cDepthHeight, m_pBackgroundRGBX_depthSpace)))
		{
			const RGBQUAD c_green = { 0, 0, 0 };

			// Fill in with a background colour of green if we can't load the background image
			for (int i = 0; i < cDepthWidth * cDepthHeight; ++i)
			{
				m_pBackgroundRGBX_depthSpace[i] = c_green;
			}
		}
	}

	// Get and initialize the default Kinect sensor
	InitializeDefaultSensor();
	Sleep(200);
	int i = 0;
	Sleep(10000); // to sync with gpu warmup
	while (1)
	{
	
		/*if (kinect_read_frame_counter > 10)
		{
			Sleep(1000);
		}*/

		Update();
	//	i++;

		//std::cout << i++ << std::endl;
		//while (!ColorImagesVec.empty()){
		//	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
		//	cv::imshow("Display window", ColorImagesVec.front());
		//	cv::waitKey(1);
		//	ColorImagesVec.pop();
		//}
		
	}

	return 1;
}

/// <summary>
/// Main processing function
/// </summary>
void CCoordinateMappingBasics::Update()
{

	if (!m_pMultiSourceFrameReader)
	{
		return;
	}

	IMultiSourceFrame* pMultiSourceFrame = NULL;
	IDepthFrame* pDepthFrame = NULL;
	IColorFrame* pColorFrame = NULL;
	IBodyIndexFrame* pBodyIndexFrame = NULL;

	Sleep(10);

	HRESULT hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

	if (SUCCEEDED(hr))
	{
		IDepthFrameReference* pDepthFrameReference = NULL;

		hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
		}

		SafeRelease(pDepthFrameReference);
	}


	if (SUCCEEDED(hr))
	{
		IColorFrameReference* pColorFrameReference = NULL;

		hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pColorFrameReference->AcquireFrame(&pColorFrame);
		}

		SafeRelease(pColorFrameReference);
	}

	if (SUCCEEDED(hr))
	{
		IBodyIndexFrameReference* pBodyIndexFrameReference = NULL;

		hr = pMultiSourceFrame->get_BodyIndexFrameReference(&pBodyIndexFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrameReference->AcquireFrame(&pBodyIndexFrame);
		}

		SafeRelease(pBodyIndexFrameReference);
	}

	if (SUCCEEDED(hr))
	{
		INT64 nDepthTime = 0;
		IFrameDescription* pDepthFrameDescription = NULL;
		int nDepthWidth = 0;
		int nDepthHeight = 0;
		UINT nDepthBufferSize = 0;
		UINT16 *pDepthBuffer = NULL;

		IFrameDescription* pColorFrameDescription = NULL;
		int nColorWidth = 0;
		int nColorHeight = 0;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		UINT nColorBufferSize = 0;
		RGBQUAD *pColorBuffer = NULL;

		IFrameDescription* pBodyIndexFrameDescription = NULL;
		int nBodyIndexWidth = 0;
		int nBodyIndexHeight = 0;
		UINT nBodyIndexBufferSize = 0;
		BYTE *pBodyIndexBuffer = NULL;

		// get depth frame data

		hr = pDepthFrame->get_RelativeTime(&nDepthTime);

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameDescription->get_Width(&nDepthWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameDescription->get_Height(&nDepthHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->AccessUnderlyingBuffer(&nDepthBufferSize, &pDepthBuffer);
		}
		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
		}

		if (SUCCEEDED(hr))
		{
			// In order to see the full range of depth (including the less reliable far field depth)
			// we are setting nDepthMaxDistance to the extreme potential depth threshold
			nDepthMaxDistance = USHRT_MAX;

			// Note:  If you wish to filter by reliable depth distance, uncomment the following line.
			//// hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxDistance);
		}

		// get color frame data

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameDescription->get_Width(&nColorWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameDescription->get_Height(&nColorHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
		}

		if (SUCCEEDED(hr))
		{
			if (imageFormat == ColorImageFormat_Bgra)
			{
				hr = pColorFrame->AccessRawUnderlyingBuffer(&nColorBufferSize, reinterpret_cast<BYTE**>(&pColorBuffer));
			}
			else if (m_pColorRGBX)
			{
				pColorBuffer = m_pColorRGBX;
				nColorBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
				hr = pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, reinterpret_cast<BYTE*>(pColorBuffer), ColorImageFormat_Bgra);
			}
			else
			{
				hr = E_FAIL;
			}
		}

		// get body index frame data

		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrame->get_FrameDescription(&pBodyIndexFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrameDescription->get_Width(&nBodyIndexWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrameDescription->get_Height(&nBodyIndexHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pBodyIndexFrame->AccessUnderlyingBuffer(&nBodyIndexBufferSize, &pBodyIndexBuffer);
		}

		if (SUCCEEDED(hr))
		{
			ProcessFrame(nDepthTime, pDepthBuffer, nDepthWidth, nDepthHeight,
				pColorBuffer, nColorWidth, nColorHeight,
				pBodyIndexBuffer, nBodyIndexWidth, nBodyIndexHeight);
		}

		SafeRelease(pDepthFrameDescription);
		SafeRelease(pColorFrameDescription);
		SafeRelease(pBodyIndexFrameDescription);
	}

	SafeRelease(pDepthFrame);
	SafeRelease(pColorFrame);
	SafeRelease(pBodyIndexFrame);
	SafeRelease(pMultiSourceFrame);
}

/// <summary>
/// Handles window messages, passes most to the class instance to handle
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CCoordinateMappingBasics::MessageRouter(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	CCoordinateMappingBasics* pThis = NULL;

	if (WM_INITDIALOG == uMsg)
	{
		pThis = reinterpret_cast<CCoordinateMappingBasics*>(lParam);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
	}
	else
	{
		pThis = reinterpret_cast<CCoordinateMappingBasics*>(::GetWindowLongPtr(hWnd, GWLP_USERDATA));
	}

	if (pThis)
	{
		return pThis->DlgProc(hWnd, uMsg, wParam, lParam);
	}

	return 0;
}

/// <summary>
/// Handle windows messages for the class instance
/// </summary>
/// <param name="hWnd">window message is for</param>
/// <param name="uMsg">message</param>
/// <param name="wParam">message data</param>
/// <param name="lParam">additional message data</param>
/// <returns>result of message processing</returns>
LRESULT CALLBACK CCoordinateMappingBasics::DlgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(wParam);
	UNREFERENCED_PARAMETER(lParam);

	switch (message)
	{
	case WM_INITDIALOG:
	{
		// Bind application window handle
		m_hWnd = hWnd;

		// Init Direct2D
		D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pD2DFactory);

		// Create and initialize a new Direct2D image renderer (take a look at ImageRenderer.h)
		// We'll use this to draw the data we receive from the Kinect to the screen
		m_pDrawCoordinateMapping = new ImageRenderer();
		HRESULT hr = m_pDrawCoordinateMapping->Initialize(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), m_pD2DFactory, cColorWidth, cColorHeight, cColorWidth * sizeof(RGBQUAD));
		if (FAILED(hr))
		{
			SetStatusMessage(L"Failed to initialize the Direct2D draw device.", 10000, true);
		}

		// Get and initialize the default Kinect sensor
		//    InitializeDefaultSensor();
	}
	break;

	// If the titlebar X is clicked, destroy app
	case WM_CLOSE:
		DestroyWindow(hWnd);
		break;

	case WM_DESTROY:
		// Quit the main message pump
		PostQuitMessage(0);
		break;

		// Handle button press
	case WM_COMMAND:
		// If it was for the screenshot control and a button clicked event, save a screenshot next frame 
		if (IDC_BUTTON_SCREENSHOT == LOWORD(wParam) && BN_CLICKED == HIWORD(wParam))
		{
			m_bSaveScreenshot = true;
		}
		break;
	}

	return FALSE;
}


/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT CCoordinateMappingBasics::InitializeDefaultSensor()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pKinectSensor)
	{
		// Initialize the Kinect and get coordinate mapper and the frame reader

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
		}


		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->OpenMultiSourceFrameReader(
				FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color | FrameSourceTypes::FrameSourceTypes_BodyIndex,
				&m_pMultiSourceFrameReader);
		}
	}

	if (!m_pKinectSensor || FAILED(hr))
	{
		SetStatusMessage(L"No ready Kinect found!", 10000, true);
		return E_FAIL;
	}

	return hr;
}

/// <summary>
/// Handle new depth and color data
/// <param name="nTime">timestamp of frame</param>
/// <param name="pDepthBuffer">pointer to depth frame data</param>
/// <param name="nDepthWidth">width (in pixels) of input depth image data</param>
/// <param name="nDepthHeight">height (in pixels) of input depth image data</param>
/// <param name="pColorBuffer">pointer to color frame data</param>
/// <param name="nColorWidth">width (in pixels) of input color image data</param>
/// <param name="nColorHeight">height (in pixels) of input color image data</param>
/// <param name="pBodyIndexBuffer">pointer to body index frame data</param>
/// <param name="nBodyIndexWidth">width (in pixels) of input body index data</param>
/// <param name="nBodyIndexHeight">height (in pixels) of input body index data</param>
/// </summary>
void CCoordinateMappingBasics::ProcessFrame(INT64 nTime,
	UINT16* pDepthBuffer, int nDepthWidth, int nDepthHeight,
	const RGBQUAD* pColorBuffer, int nColorWidth, int nColorHeight,
	const BYTE* pBodyIndexBuffer, int nBodyIndexWidth, int nBodyIndexHeight)
{

	// Make sure we've received valid data
	if (m_pCoordinateMapper && m_pDepthCoordinates && m_pOutputRGBX &&
		pDepthBuffer && (nDepthWidth == cDepthWidth) && (nDepthHeight == cDepthHeight) &&
		pColorBuffer && (nColorWidth == cColorWidth) && (nColorHeight == cColorHeight) &&
		pBodyIndexBuffer && (nBodyIndexWidth == cDepthWidth) && (nBodyIndexHeight == cDepthHeight))
	{

		cv::Mat DeptImg(nDepthHeight, nDepthWidth, CV_16UC1, reinterpret_cast<BYTE *>(pDepthBuffer));

		RGBQUAD* pRGBX = m_pDepthRGBX;
		RGBQUAD* pRGBX_mask = m_pOutputRGBX_depthSpace;
		// end pixel is start + width*height - 1
		const UINT16* pBufferEnd = pDepthBuffer + (nDepthWidth * nDepthHeight);

		//  HRESULT hr = m_pCoordinateMapper->MapColorFrameToDepthSpace(nDepthWidth * nDepthHeight, (UINT16*)pDepthBuffer, nColorWidth * nColorHeight, m_pDepthCoordinates);
		HRESULT hr = m_pCoordinateMapper->MapDepthFrameToColorSpace(nDepthWidth * nDepthHeight, (UINT16*)pDepthBuffer, nDepthWidth * nDepthHeight, m_pColorCoordinates);
		//CameraIntrinsics* cam;
		//	m_pCoordinateMapper->GetDepthCameraIntrinsics(cam);

		int depthIndexTemp = 0;
		while (pDepthBuffer < pBufferEnd)
		{

			USHORT depth = *pDepthBuffer;

			// To convert to a byte, we're discarding the most-significant
			// rather than least-significant bits.
			// We're preserving detail, although the intensity will "wrap."
			// Values outside the reliable depth range are mapped to 0 (black).

			// Note: Using conditionals in this loop could degrade performance.
			// Consider using a lookup table instead when writing production code.
			BYTE intensity = static_cast<BYTE>((depth >= nDepthMinReliableDistance) && (depth <= nDepthMaxDistance) ? (depth % 256) : 0);

			pRGBX->rgbRed = intensity;
			pRGBX->rgbGreen = intensity;
			pRGBX->rgbBlue = intensity;

			//pRGBX_mask->rgbRed = static_cast<BYTE>(0);
			//	pRGBX_mask->rgbGreen = static_cast<BYTE>(0);
			//	pRGBX_mask->rgbBlue = static_cast<BYTE>(0);
			BYTE player = pBodyIndexBuffer[depthIndexTemp];
			if (player != 0xff){
				pRGBX_mask->rgbRed = static_cast<BYTE>(255);
				pRGBX_mask->rgbGreen = static_cast<BYTE>(255);
				pRGBX_mask->rgbBlue = static_cast<BYTE>(255);

			}
			else{

				pRGBX_mask->rgbRed = static_cast<BYTE>(0);
				pRGBX_mask->rgbGreen = static_cast<BYTE>(0);
				pRGBX_mask->rgbBlue = static_cast<BYTE>(0);

			}

			++pRGBX;
			++pRGBX_mask;
			++pDepthBuffer;
			++depthIndexTemp;
		}

		if (SUCCEEDED(hr))
		{
			/* loop over output pixel in detph space*/
			//RGBQUAD* pRGBX;
			for (int depthIndex = 0; depthIndex < (nDepthWidth*nDepthHeight); ++depthIndex)
			{
				RGBQUAD* pSrc = m_pBackgroundRGBX_depthSpace + depthIndex;
				ColorSpacePoint p = m_pColorCoordinates[depthIndex];


				//	if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity())
				{
					int depthX = static_cast<int>(p.X + 0.5f);
					int depthY = static_cast<int>(p.Y + 0.5f);

					if ((depthX >= 0 && depthX < nColorWidth) && (depthY >= 0 && depthY < nColorHeight))
					{
						//BYTE player = pBodyIndexBuffer[depthIndex];

						// if we're tracking a player for the current pixel, draw from the color camera
						//if (player != 0xff)
						{
							// set source for copy to the color pixel
							pSrc = m_pColorRGBX + depthX + depthY*nColorWidth;
						}
					}
					//		pSrc->rgbRed = static_cast<BYTE>(1);
					//		pSrc->rgbGreen = static_cast<BYTE>(0);
					//		pSrc->rgbBlue = static_cast<BYTE>(0);


					m_pBackgroundRGBX_depthSpace[depthIndex] = *pSrc;
				}
			}

			// save images
			m_bSaveScreenshot = true;

			if (m_bSaveScreenshot)
			{
				//WCHAR szScreenshotPath[MAX_PATH];

				//// Retrieve the path to My Photos
				//GetScreenshotFileName(szScreenshotPath, _countof(szScreenshotPath));

				
				// save color

				//wchar_t m_reportFileName[256];

			//	swprintf_s(m_reportFileName, L"%04d", frame_count);
			//	StringCchPrintf(szScreenshotPath, _countof(szScreenshotPath), L"C:\\trackingData\\Data\\color\\frame_%s.bmp", m_reportFileName);
				//	hr = SaveBitmapToFile(reinterpret_cast<BYTE*>(m_pColorRGBX), nColorWidth, nColorHeight, sizeof(RGBQUAD) * 8, szScreenshotPath);

				// Write out the bitmap to disk
			//	hr = SaveBitmapToFile(reinterpret_cast<BYTE*>(m_pBackgroundRGBX_depthSpace), nDepthWidth, nDepthHeight, sizeof(RGBQUAD) * 8, szScreenshotPath);
				cv::Mat colorImg(nDepthHeight, nDepthWidth, CV_8UC4, reinterpret_cast<BYTE *>(m_pBackgroundRGBX_depthSpace));
		
				/*char colorName[128] = "C:\\trackingData\\Data\\color\\frame_%04d_color.png";
				char colorFullName[128];
				sprintf(colorFullName, colorName, kinect_read_frame_counter);
				cv::imwrite(colorFullName, colorImg);*/

				ColorImagesQueue.push_back(colorImg.clone()); // push color image to golobal buffer
			
				// save depth 
	

				//WCHAR szScreenshotPathDepth[MAX_PATH];
				//StringCchPrintf(szScreenshotPathDepth, _countof(szScreenshotPathDepth), L"C:\\trackingData\\Data\\depth\\frame_%s_depth.bmp", m_reportFileName);
				//// Write out the bitmap to disk
				////hr = SaveBitmapToFile(reinterpret_cast<BYTE*>(m_pDepthRGBX), nDepthWidth, nDepthHeight, sizeof(RGBQUAD) * 8, szScreenshotPathDepth);
				//char aName[128] = "C:\\trackingData\\Data\\depth\\frame_%04d_depth.pgm";
				//char aNameDepth[128];
				//sprintf(aNameDepth, aName, kinect_read_frame_counter);
				//cv::imwrite(aNameDepth, DeptImg);

				DeptImg.convertTo(DeptImg, CV_32FC1);
				depthImagesQueue.push_back(DeptImg.clone());  // push depth image to golobal buffer


				// save mask
				//WCHAR szScreenshotPathMask[MAX_PATH];
				//StringCchPrintf(szScreenshotPathMask, _countof(szScreenshotPathMask), L"C:\\trackingData\\Data\\mask\\frame_%s_mask.bmp", m_reportFileName);
				//// Write out the bitmap to disk
				//hr = SaveBitmapToFile(reinterpret_cast<BYTE*>(m_pOutputRGBX_depthSpace), nDepthWidth, nDepthHeight, sizeof(RGBQUAD) * 8, szScreenshotPathMask);

				cv::Mat MaskImage(nDepthHeight, nDepthWidth, CV_8UC4, reinterpret_cast<BYTE *>(m_pOutputRGBX_depthSpace));
				maskImagesQueue.push_back(MaskImage.clone());


				kinect_read_frame_counter++;

				//    WCHAR szStatusMessage[64 + MAX_PATH];
				//    if (SUCCEEDED(hr))
				//    {
				//        // Set the status bar to show where the screenshot was saved
				//        StringCchPrintf(szStatusMessage, _countof(szStatusMessage), L"Screenshot saved to %s", szScreenshotPath);
				//     }
				//     else
				//     {
				//         StringCchPrintf(szStatusMessage, _countof(szStatusMessage), L"Failed to write screenshot to %s", szScreenshotPath);
				//     }

				//     SetStatusMessage(szStatusMessage, 5000, true);

				// toggle off so we don't save a screenshot again next frame
				// m_bSaveScreenshot = false;
			}
		}
	}
}

/// <summary>
/// Set the status bar message
/// </summary>
/// <param name="szMessage">message to display</param>
/// <param name="showTimeMsec">time in milliseconds to ignore future status messages</param>
/// <param name="bForce">force status update</param>
bool CCoordinateMappingBasics::SetStatusMessage(_In_z_ WCHAR* szMessage, DWORD nShowTimeMsec, bool bForce)
{
	INT64 now = GetTickCount64();

	if (m_hWnd && (bForce || (m_nNextStatusTime <= now)))
	{
		SetDlgItemText(m_hWnd, IDC_STATUS, szMessage);
		m_nNextStatusTime = now + nShowTimeMsec;

		return true;
	}

	return false;
}

/// <summary>
/// Get the name of the file where screenshot will be stored.
/// </summary>
/// <param name="lpszFilePath">string buffer that will receive screenshot file name.</param>
/// <param name="nFilePathSize">number of characters in lpszFilePath string buffer.</param>
/// <returns>
/// S_OK on success, otherwise failure code.
/// </returns>
HRESULT CCoordinateMappingBasics::GetScreenshotFileName(_Out_writes_z_(nFilePathSize) LPWSTR lpszFilePath, UINT nFilePathSize)
{
	WCHAR* pszKnownPath = NULL;
	HRESULT hr = SHGetKnownFolderPath(FOLDERID_Pictures, 0, NULL, &pszKnownPath);

	if (SUCCEEDED(hr))
	{
		// Get the time
		WCHAR szTimeString[MAX_PATH];
		GetTimeFormatEx(NULL, 0, NULL, L"hh'-'mm'-'ss", szTimeString, _countof(szTimeString));

		// File name will be KinectScreenshotDepth-HH-MM-SS.bmp
		StringCchPrintfW(lpszFilePath, nFilePathSize, L"%s\\KinectScreenshot-CoordinateMapping-%s.bmp", pszKnownPath, szTimeString);
	}

	if (pszKnownPath)
	{
		CoTaskMemFree(pszKnownPath);
	}

	return hr;
}

/// <summary>
/// Save passed in image data to disk as a bitmap
/// </summary>
/// <param name="pBitmapBits">image data to save</param>
/// <param name="lWidth">width (in pixels) of input image data</param>
/// <param name="lHeight">height (in pixels) of input image data</param>
/// <param name="wBitsPerPixel">bits per pixel of image data</param>
/// <param name="lpszFilePath">full file path to output bitmap to</param>
/// <returns>indicates success or failure</returns>
HRESULT CCoordinateMappingBasics::SaveBitmapToFile(BYTE* pBitmapBits, LONG lWidth, LONG lHeight, WORD wBitsPerPixel, LPCWSTR lpszFilePath)
{
	DWORD dwByteCount = lWidth * lHeight * (wBitsPerPixel / 8);

	BITMAPINFOHEADER bmpInfoHeader = { 0 };

	bmpInfoHeader.biSize = sizeof(BITMAPINFOHEADER);  // Size of the header
	bmpInfoHeader.biBitCount = wBitsPerPixel;             // Bit count
	bmpInfoHeader.biCompression = BI_RGB;                    // Standard RGB, no compression
	bmpInfoHeader.biWidth = lWidth;                    // Width in pixels
	bmpInfoHeader.biHeight = -lHeight;                  // Height in pixels, negative indicates it's stored right-side-up
	bmpInfoHeader.biPlanes = 1;                         // Default
	bmpInfoHeader.biSizeImage = dwByteCount;               // Image size in bytes

	BITMAPFILEHEADER bfh = { 0 };

	bfh.bfType = 0x4D42;                                           // 'M''B', indicates bitmap
	bfh.bfOffBits = bmpInfoHeader.biSize + sizeof(BITMAPFILEHEADER);  // Offset to the start of pixel data
	bfh.bfSize = bfh.bfOffBits + bmpInfoHeader.biSizeImage;        // Size of image + headers

	// Create the file on disk to write to
	HANDLE hFile = CreateFileW(lpszFilePath, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	// Return if error opening file
	if (NULL == hFile)
	{
		return E_ACCESSDENIED;
	}

	DWORD dwBytesWritten = 0;

	// Write the bitmap file header
	if (!WriteFile(hFile, &bfh, sizeof(bfh), &dwBytesWritten, NULL))
	{
		CloseHandle(hFile);
		return E_FAIL;
	}

	// Write the bitmap info header
	if (!WriteFile(hFile, &bmpInfoHeader, sizeof(bmpInfoHeader), &dwBytesWritten, NULL))
	{
		CloseHandle(hFile);
		return E_FAIL;
	}

	// Write the RGB Data
	if (!WriteFile(hFile, pBitmapBits, bmpInfoHeader.biSizeImage, &dwBytesWritten, NULL))
	{
		CloseHandle(hFile);
		return E_FAIL;
	}

	// Close the file
	CloseHandle(hFile);
	return S_OK;
}

/// <summary>
/// Load an image from a resource into a buffer
/// </summary>
/// <param name="resourceName">name of image resource to load</param>
/// <param name="resourceType">type of resource to load</param>
/// <param name="nOutputWidth">width (in pixels) of scaled output bitmap</param>
/// <param name="nOutputHeight">height (in pixels) of scaled output bitmap</param>
/// <param name="pOutputBuffer">buffer that will hold the loaded image</param>
/// <returns>S_OK on success, otherwise failure code</returns>
HRESULT CCoordinateMappingBasics::LoadResourceImage(PCWSTR resourceName, PCWSTR resourceType, UINT nOutputWidth, UINT nOutputHeight, RGBQUAD* pOutputBuffer)
{
	IWICImagingFactory* pIWICFactory = NULL;
	IWICBitmapDecoder* pDecoder = NULL;
	IWICBitmapFrameDecode* pSource = NULL;
	IWICStream* pStream = NULL;
	IWICFormatConverter* pConverter = NULL;
	IWICBitmapScaler* pScaler = NULL;

	HRSRC imageResHandle = NULL;
	HGLOBAL imageResDataHandle = NULL;
	void *pImageFile = NULL;
	DWORD imageFileSize = 0;

	HRESULT hrCoInit = CoInitialize(NULL);
	HRESULT hr = hrCoInit;

	if (SUCCEEDED(hr))
	{
		hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory, (LPVOID*)&pIWICFactory);
	}

	if (SUCCEEDED(hr))
	{
		// Locate the resource
		imageResHandle = FindResourceW(HINST_THISCOMPONENT, resourceName, resourceType);
		hr = imageResHandle ? S_OK : E_FAIL;
	}

	if (SUCCEEDED(hr))
	{
		// Load the resource
		imageResDataHandle = LoadResource(HINST_THISCOMPONENT, imageResHandle);
		hr = imageResDataHandle ? S_OK : E_FAIL;
	}

	if (SUCCEEDED(hr))
	{
		// Lock it to get a system memory pointer.
		pImageFile = LockResource(imageResDataHandle);
		hr = pImageFile ? S_OK : E_FAIL;
	}

	if (SUCCEEDED(hr))
	{
		// Calculate the size.
		imageFileSize = SizeofResource(HINST_THISCOMPONENT, imageResHandle);
		hr = imageFileSize ? S_OK : E_FAIL;
	}

	if (SUCCEEDED(hr))
	{
		// Create a WIC stream to map onto the memory.
		hr = pIWICFactory->CreateStream(&pStream);
	}

	if (SUCCEEDED(hr))
	{
		// Initialize the stream with the memory pointer and size.
		hr = pStream->InitializeFromMemory(
			reinterpret_cast<BYTE*>(pImageFile),
			imageFileSize);
	}

	if (SUCCEEDED(hr))
	{
		// Create a decoder for the stream.
		hr = pIWICFactory->CreateDecoderFromStream(
			pStream,
			NULL,
			WICDecodeMetadataCacheOnLoad,
			&pDecoder);
	}

	if (SUCCEEDED(hr))
	{
		// Create the initial frame.
		hr = pDecoder->GetFrame(0, &pSource);
	}

	if (SUCCEEDED(hr))
	{
		// Convert the image format to 32bppPBGRA
		// (DXGI_FORMAT_B8G8R8A8_UNORM + D2D1_ALPHA_MODE_PREMULTIPLIED).
		hr = pIWICFactory->CreateFormatConverter(&pConverter);
	}

	if (SUCCEEDED(hr))
	{
		hr = pIWICFactory->CreateBitmapScaler(&pScaler);
	}

	if (SUCCEEDED(hr))
	{
		hr = pScaler->Initialize(
			pSource,
			nOutputWidth,
			nOutputHeight,
			WICBitmapInterpolationModeCubic
			);
	}

	if (SUCCEEDED(hr))
	{
		hr = pConverter->Initialize(
			pScaler,
			GUID_WICPixelFormat32bppPBGRA,
			WICBitmapDitherTypeNone,
			NULL,
			0.f,
			WICBitmapPaletteTypeMedianCut);
	}

	UINT width = 0;
	UINT height = 0;
	if (SUCCEEDED(hr))
	{
		hr = pConverter->GetSize(&width, &height);
	}

	// make sure the image scaled correctly so the output buffer is big enough
	if (SUCCEEDED(hr))
	{
		if ((width != nOutputWidth) || (height != nOutputHeight))
		{
			hr = E_FAIL;
		}
	}

	if (SUCCEEDED(hr))
	{
		hr = pConverter->CopyPixels(NULL, width * sizeof(RGBQUAD), nOutputWidth * nOutputHeight * sizeof(RGBQUAD), reinterpret_cast<BYTE*>(pOutputBuffer));
	}

	SafeRelease(pScaler);
	SafeRelease(pConverter);
	SafeRelease(pSource);
	SafeRelease(pDecoder);
	SafeRelease(pStream);
	SafeRelease(pIWICFactory);

	if (SUCCEEDED(hrCoInit))
	{
		CoUninitialize();
	}

	return hr;
}

