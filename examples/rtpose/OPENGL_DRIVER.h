
//  Class OPENGL_DRIVER
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __OPENGL_DRIVER_H__
#define __OPENGL_DRIVER_H__
#include <GL/glut.h>
//#include "ARMADILLO.h"
#include "../lib/MESH.h"
//#include "HumanBody.h"
//#include "landmarks.h"
//#include "skeleton.h"
//#include "geodesic_algorithm_exact.h"
#include <fstream>
//#include "3DplaneFitting.h"

#include "Depth2Mesh.h"

//#define SUB11 1
//#define SUB12 2
//#define SUB13 3
//#define SUB14 4


#define NO_MOTION			0
#define ZOOM_MOTION			1
#define ROTATE_MOTION		2
#define TRANSLATE_MOTION	3
//#define	FLOAT_TYPE	double
//bool	idle_run=false;
//int		file_id=0;
//double	time_step=1/30.0;
//int x_start, y_start;
//int x_end, y_end;
//
int joints_links[17][2] = { { 16, 14 }, { 15, 17 },{ 14, 0 }, { 0, 15 },  { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 },
{ 1, 8 }, { 8, 9 }, { 9, 10 }, { 1, 11 }, { 11, 12 }, { 12, 13 }, { 1, 5 }, {5, 6 }, {6,7} };
//double SphereCenter_old[72][3];
//
//int		select_v=-1;
//int		select_old_v = -1;
//int select_new_v = -1;
//
//int		select_target_v = -1;
//int   select_target_old_v = -1;
//
//int left_button_flag = 0;
//int select_landmarks_v = -1;
//double	target[3];
//int geodesic_select_trigger = 0;
//int geodesic_calculate_trigger = 0;
//int subMenu1;
//int mainMenu;
//int FunctionMode = 0;
//geodesic::Mesh GeoMesh;
//
//std::vector<geodesic::SurfacePoint> path;	//geodesic path is a sequence of SurfacePoints
/////////////////////////////////////////////////////////////////////////////////////////////
//




int		screen_width = 1024;
int        screen_height = 768;
int		mesh_mode = 0;
int		render_mode = 0;
int        geodesicDistance_mode = 0;
double	    zoom = 30;
double     swing_angle = -0;
double     elevate_angle = 0;
double     z_angle = -0;
static double	    center[3] = { 0, 0, 0 };
int		motion_mode = NO_MOTION;
int        mouse_x = 0;
int        mouse_y = 0;

bool	 idle_run = true;
double	 time_step = 1 / 30.0;
static  int frame_id = 5;



template <class TYPE>
void Update(TYPE t)
{
////	armadillo.Update(t, 64, select_v, target);
}


///////////////////////////////////////////////////////////////////////////////////////////
//  class OPENGL_DRIVER
///////////////////////////////////////////////////////////////////////////////////////////
class OPENGL_DRIVER
{
public:
	 int		file_id=0;

	//3D display configuration.
	/* int		screen_width = 1024;
	 int        screen_height=768;
	 int		mesh_mode=0;
	 int		render_mode=0;
	 int        geodesicDistance_mode=0;
	 double	    zoom = 30;
	 double     swing_angle = -0;
	 double     elevate_angle = 0;
	 double     z_angle=-0;
	 double	    center[3] = {0,0,0};
	 int		motion_mode = NO_MOTION;
	 int        mouse_x = 0;
	 int        mouse_y=0;*/

	static  MESH<FLOAT_TYPE>* mesh;

	//bool	 idle_run = false;
	//double	 time_step = 1 / 30.0;
	int      x_start, y_start;
	int       x_end, y_end;
	//static MESH<FLOAT_TYPE>* mesh_landmarks;
	//static MESH<FLOAT_TYPE>* mesh_skeleton;
	//static double** SphereCenter;
	//static double** joints;
	//static int** joints_links;
	//static double** SphereCenter_origin;
	//static cv::Mat* TextureImage;
//	static GLubyte*** textBMP; 
	static GLuint texName;
	static vector<string> txtfileNames;
	// std::vector<double> points;
	// std::vector<unsigned> faces;
	
	static vector<Mat> textureImages;
	static vector<Mat> depthImages;
	static vector<Mat> maskImages;
	static vector<vector<Point2f>> joints_all;

	//OPENGL_DRIVER(MESH<FLOAT_TYPE> &mesh_new)
	OPENGL_DRIVER()
	{

		//mesh = &mesh_new;

		


		////////////////// get the file list

		// find all the txt filenames
		
		WIN32_FIND_DATA fd;
		string txtFolder = "E:/skeletonTracking/jointsLocation/*.txt";

		HANDLE hFind = FindFirstFile(txtFolder.c_str(), &fd);
		if (hFind != INVALID_HANDLE_VALUE) {
			do {
				// read all (real) files in current folder
				// , delete '!' read other 2 default folder . and ..
				if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
					txtfileNames.push_back(fd.cFileName);
				}
			} while (FindNextFile(hFind, &fd));
			FindClose(hFind);
		}

		int valid_frame_num = static_cast<int>(txtfileNames.size());

		// read all the data to vectors
		for (int i = 0; i <250; i++)
		{
			string jsonName = txtfileNames[i];
			size_t f_inx = jsonName.find_first_of("0");
			string frameIndex = jsonName.substr(f_inx + 2, 4);
			string frameName = "frame_" + frameIndex;

			string textureName = "E:/skeletonTracking/Data2/color/" + frameName + ".bmp";
			string depthName = "E:/skeletonTracking/Data2/depth/" + frameName + "_depth.pgm";
			string maskName = "E:/skeletonTracking/Data2/mask/" + frameName + "_mask.bmp";


			// load the texture image
			//	Mat textureImage = imread("E:\\skeletonTracking\\Data2\\color\\frame_0052.bmp", CV_LOAD_IMAGE_COLOR);   // Read the file
			//Mat textureImage = imread(textureName, CV_LOAD_IMAGE_COLOR);   // Read the file

		//	if (!textureImage.data)                              // Check for invalid input
		//	{
		//		cout << "Could not open or find the image" << std::endl;
				//return -1;
		//	}

		//	textureImage.convertTo(textureImage, CV_32FC3);
		//	textureImage = textureImage / 255.0f;

		//	textureImages.push_back(textureImage);

			// load the depth image  bn	6  
			//	Mat depth = imread("E:\\skeletonTracking\\Data2\\depth\\frame_0052_depth.pgm", CV_LOAD_IMAGE_ANYDEPTH);   // Read the file
			Mat depth = imread(depthName, CV_LOAD_IMAGE_ANYDEPTH);   // Read the file

			if (!depth.data)                              // Check for invalid input
			{
				cout << "Could not open or find the image" << std::endl;
				//return -1;
			}
			Mat depth_origin = depth.clone();
			depth.convertTo(depth, CV_32FC1);

			depthImages.push_back(depth);

			//	cout << depth.type() << endl;
			//depth = depth / 255.0f;

			// load the mask image 
			//	Mat mask_origin = imread("E:\\skeletonTracking\\Data2\\mask\\frame_0052_mask.bmp");   // Read the file
			Mat mask_origin = imread(maskName);   // Read the file

			if (!mask_origin.data)                              // Check for invalid input
			{
				cout << "Could not open or find the image" << std::endl;
				//return -1;
			}

			maskImages.push_back(mask_origin);
			vector<Point2f> joints;
			ifstream inputfile("E:/skeletonTracking/jointsLocation/" + jsonName);

			
			if (inputfile){
				float x, y, t;
				char comma;

				while (inputfile >> x){
					inputfile >> comma;
					inputfile >> y;
					inputfile >> comma;
					inputfile >> t;
					joints.push_back(Point2f(x, y));

				}

			}

			inputfile.close();

			joints_all.push_back(joints);

		}
		  

		



		//std::cout << mesh->VT[0] <<std::endl;
		int argc = 1;
		char *argv[1] = { (char*)"Something" };
		glutInit(&argc, argv);
		
		glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
		glutInitWindowPosition(50,50);
		glutInitWindowSize(screen_width, screen_height);
		glutCreateWindow ("Human Body Cloth");
		glutDisplayFunc(Handle_Display);
		glutReshapeFunc(Handle_Reshape);
		//glutKeyboardFunc(Handle_Keypress);
		//glutMouseFunc(Handle_Mouse_Click);
		//glutMotionFunc(Handle_Mouse_Move);
		//glutSpecialFunc(Handle_SpecialKeypress);
		glutIdleFunc(Handle_Idle);
		Handle_Reshape(screen_width, screen_height);
		
	//////////////init texture bmp
		
		/*if (TextureImage != 0 && texture_flag !=0)
		{
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, (TextureImage->step & 3) ? 1 : 4);
		glPixelStorei(GL_PACK_ROW_LENGTH, TextureImage->step / TextureImage->elemSize());

		glGenTextures(1, &texName);
		glBindTexture(GL_TEXTURE_2D, texName);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
		GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
		GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TextureImage->cols,
		TextureImage->rows, 0, GL_RGB, GL_UNSIGNED_BYTE,
		TextureImage->data);

		}*/
		////////////////////////

		//initializeMenus();
	//	textureImages.pop_back();
		glutMainLoop();
		
	}

	

	 static void Handle_Display()
	{	

	
		glLoadIdentity();
		glClearColor(0.8,0.8,0.8,0);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
						
		gluLookAt(0, 0, zoom, 0, 0, 0, 0, 1, 0);
		//glRotated(elevate_angle, 1, 0, 0);
		
		//glRotated(swing_angle, 0, 1, 0);

	
		glRotated(elevate_angle, 1, 0, 0);

		glRotated(swing_angle, 0, 1, 0);
		glRotated(z_angle, 0, 0, -1);
		glTranslatef(-center[0], -center[1], -center[2]);
		
		int flag_texture = 0;
		/*int flag_bodyOrLandmarks = 1;
		int flag_texture = 0;
		if (texture_flag  == 0){
		
			flag_texture = 0;
			
		}
		else{
			flag_texture = 1;
			


		}*/

		
		//cout << mesh->t_number << endl;


		///////////////////////////////////////////////////////
		////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////
		



		// load jason loactions
		clock_t begin = clock();
		Mat depth = depthImages[frame_id];
		Mat mask_origin = maskImages[frame_id];
		vector<Point2f> joints = joints_all[frame_id];

		// Begin to convert
		Mat pointmap = depthmap2pointmap(depth);

		Mat mask = im2bw(mask_origin, 125.0);
		cv::cvtColor(mask, mask, CV_BGR2GRAY);
		//cout << mask.type() << endl;
		                                                         

		Mat mask_new = MaskBlob(mask);
		mask_new = im2bw(mask_new, 1.0);

	
		MESH<FLOAT_TYPE> mesh_new;
		pointmap2mesh(pointmap, mask_new, mesh_new);
		//////////////////////////////////////RemoveLong Face


		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		cout << "time elipse :" << elapsed_secs << endl;
	
		////////////////////////////////////update texture 

		// mesh.Write_OBJ("test5.obj");



		// convert skeleton

		//blur(depth, depth, Size(5, 5));

	

		vector<Point3f> joints_dst(joints.size());
		Mat temp = Mat(1, 3, CV_32FC1);
		Mat KK_t = KK.t();
		for (int i = 0; i < joints.size(); i++)
		{
			temp.at<float>(0, 2) = depth.at<float>(round(joints[i].y), round(joints[i].x));
			temp.at<float>(0, 0) = joints[i].x* temp.at<float>(0, 2);
			temp.at<float>(0, 1) = joints[i].y* temp.at<float>(0, 2);



			Mat result = temp*KK_t.inv();
			joints_dst[i].x = result.at<float>(0, 0);
			joints_dst[i].y = result.at<float>(0, 1);
			joints_dst[i].z = result.at<float>(0, 2);
		}
		
		//cout << joints_dst << endl;

		// unproject2DPoints
		Mat proj = KK*Rt;

		Mat Mat_proj_13;
		Mat Mat_proj_t;
		proj(Rect(0, 0, 3, 3)).copyTo(Mat_proj_13);
		proj(Rect(3, 0, 1, 3)).copyTo(Mat_proj_t);

		Mat cop = -Mat_proj_13.inv()*Mat_proj_t;

		Mat joints_3d = Mat(joints.size(), 3, CV_32FC1);
		for (int i = 0; i < joints.size(); i++)
		{
			joints_3d.at<float>(i, 0) = joints[i].x;
			joints_3d.at<float>(i, 1) = joints[i].y;
			joints_3d.at<float>(i, 2) = 1;
		}

		Mat dir_temp = Mat_proj_13.inv()*joints_3d.t();
		Mat dir = dir_temp.t();


		// compute the intersection points
		float p[3];
		p[0] = cop.at<float>(0, 0);
		p[1] = cop.at<float>(1, 0);
		p[2] = cop.at<float>(2, 0);
		// mesh.Scale(1000);

		vector<Point3f> joints_dst2(joints.size(), Point3f(0, 0, 0));
		vector<int> intersecetion_ind(joints.size());
		for (int i = 0; i < dir.rows; i++)
		{
			float q[3];
			q[1] = p[1] + dir.at<float>(i, 1);
			q[0] = p[0] + dir.at<float>(i, 0);
			q[2] = p[2] + dir.at<float>(i, 2);
			int temp_select_v = -1;


			mesh_new.Select(p, q, temp_select_v);
		//	cout << temp_select_v << endl;
			intersecetion_ind[i] = temp_select_v;
			if (temp_select_v != -1)
			{
				joints_dst2[i].x = mesh_new.X[temp_select_v * 3 + 0];
				joints_dst2[i].y = mesh_new.X[temp_select_v * 3 + 1];
				joints_dst2[i].z = mesh_new.X[temp_select_v * 3 + 2];
			}

		}

		
		vector<Point3f> joints_dst3(joints.size());

		for (int i = 0; i < joints.size(); i++)
		{

			if (intersecetion_ind[i] != -1){
				joints_dst3[i].x = (joints_dst[i].x + joints_dst2[i].x) / 2;
				joints_dst3[i].y = (joints_dst[i].y + joints_dst2[i].y) / 2;
				joints_dst3[i].z = (joints_dst[i].z + joints_dst2[i].z) / 2;
			}
			else{


				joints_dst3[i].x = (joints_dst[i].x);
				joints_dst3[i].y = (joints_dst[i].y);
				joints_dst3[i].z = (joints_dst[i].z);




			}
		

			mesh_new.Make_Sphere(joints_dst3[i].x, joints_dst3[i].y, joints_dst3[i].z, 14, 20, 20);
		}

	//	cout << mesh_new.number << endl;

		mesh_new.Scale(0.001);
		//OPENGL_DRIVER myRender(mesh);

		FLOAT_TYPE min_x, min_y, min_z;
		FLOAT_TYPE max_x, max_y, max_z;
		mesh_new.Range(min_x, min_y, min_z, max_x, max_y, max_z);

		double size_x = (max_x + min_x) / 2.0;
		double size_y = (max_y + min_y) / 2.0;
		double size_z = (max_z + min_z) / 2.0;

		if (frame_id == 5){
			center[0] = 0;// size_x;
			center[1] = 0;// size_y;
			center[2] = 0;// size_z;
		}

		//mesh->Center();

		//Scale(0.008);
		//mesh_new.Centralize();

		//string obj_filename = "D:\\yajieCOde\\BodyMeasurementProject\\RenderToDepth\\renderToMesh\\renderToMesh\\tempData\\" + txtfileNames[frame_id] + ".obj";

		//mesh_new.Write_OBJ(obj_filename.c_str());
		mesh = &mesh_new;

























		/////////////////////////////////////////////////////////////////



	


	//	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	//	moveWindow("Display window", 1000, 20);
	//	imshow("Display window", textureImages[frame_id]);

	//	waitKey(1);

		
		mesh->Render(render_mode,1, flag_texture);
		
		
		

		for (int i = 0; i < joints.size()-2; i++){
			glPushMatrix();
			glColor3f(0, 1, 0);
			
			glTranslatef(joints_dst3[i].x/1000, -joints_dst3[i].y/1000,joints_dst3[i].z/1000+0.1);
			glutSolidSphere(0.008, 20, 20);
			glPopMatrix();

		}
		

		// render SphereCenter

	/*	glDisable(GL_LIGHTING);
		glColor3f(1, 0, 0);
		glTranslatef(SphereCenter[i][0], SphereCenter[i][1], SphereCenter[i][2]);
		glutSolidSphere(0.008, 10, 10);
		glPopMatrix();
		glEnable(GL_LIGHTING);*/





		// draw links

		for (int i = 2; i < 17; i++){

			//glPushMatrix();
			//glColor3f(0, 0, 1);
			//glTranslatef(mesh->X[select_target_v * 3 + 0], mesh->X[select_target_v * 3 + 1], mesh->X[select_target_v * 3 + 2]);
			//glutSolidSphere(0.005, 10, 10);
			//glPopMatrix();


		
		//	glDisable(GL_DEPTH_TEST);
			glColor3f(1, 0, 1);
			glBegin(GL_LINES);

			glLineWidth(200.0);
			glVertex3f(joints_dst3[joints_links[i][0]].x/1000, -joints_dst3[joints_links[i][0]].y/1000, joints_dst3[joints_links[i][0]].z/1000+0.1);
			glVertex3f(joints_dst3[joints_links[i][1]].x/1000,- joints_dst3[joints_links[i][1]].y/1000, joints_dst3[joints_links[i][1]].z/1000+0.1);
			glEnd();
		//	glEnable(GL_DEPTH_TEST);
			
		}
		



			
		
		
		glutSwapBuffers();
	}

	 static void Handle_Idle()
	{
		if(idle_run)	Update(time_step);

		frame_id = ((frame_id+1) % 250);
		
		glutPostRedisplay();
	}
	//menu call back function

	 void processMenu(int option)
	{
		


	}

	 void initializeMenus(void)
	{
		//subMenu1 = glutCreateMenu(processMenu);
		//glutAddMenuEntry("Landmarks Correction", SUB11);
		//glutAddMenuEntry("Geodesic Distance Between A and B", SUB12);
		//glutAddMenuEntry("Body Measurement", SUB13);
		//glutAddMenuEntry("Reset to Start", SUB14);
	//	mainMenu = glutCreateMenu(processMenu);
	//	glutAddSubMenu("Mode Selection", subMenu1);
	//	glutAttachMenu(GLUT_RIGHT_BUTTON);

	}
	static void Handle_Reshape(int w,int h)
	{
		screen_width=w,screen_height=h;
		glViewport(0,0,w, h);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(4, (double)w/(double)h, 1, 100);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glEnable(GL_DEPTH_TEST);
		{
			GLfloat LightDiffuse[] = { 1.0, 1.0, 1.0, 1};
			GLfloat LightPosition[] = { 0, 0, -100};
			glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);
			glLightfv(GL_LIGHT0, GL_POSITION,LightPosition);
			glEnable(GL_LIGHT0);
		}			
		{
			GLfloat LightDiffuse[] = { 1.0, 1.0, 1.0, 1};
			GLfloat LightPosition[] = { 0, 0, 100};
			glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
			glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);
			glEnable(GL_LIGHT1);
		}
		glEnable(GL_LIGHTING);
		glShadeModel(GL_SMOOTH);		
		glutPostRedisplay();
	}
	
	static void Handle_Keypress(unsigned char key,int mousex,int mousey)
	{
		switch(key)
		{
			case 27: exit(0);
			case 'a':
			{ 
				zoom-=2; 
				if(zoom<0.3) zoom=0.3; 
				break;
			}
			case 'z':
			{
				zoom+=2; 
				break;
			}
			case 't':
			{
				render_mode=(render_mode+1)%4;
				break;
			}
			case 'g':{

			//	geodesicDistance_mode = (geodesicDistance_mode+1)%2;
				break;



				}
			case 's':
			{
				
				break;
			}
			case 'd':
			{
			//	geodesic_select_trigger = (geodesic_select_trigger + 1) % 2;

				break;
			}
			case '1':
			{
				idle_run=true;
				break;
			}
			
			
			
		}
		glutPostRedisplay();
	}

	template <class TYPE>
	 void Get_Selection_Ray(int mouse_x, int mouse_y, TYPE* p, TYPE* q)
	{
		// Convert (x, y) into the 2D unit space
		double new_x = (double)(2*mouse_x)/(double)screen_width-1;
		double new_y = 1-(double)(2*mouse_y)/(double)screen_height;

		// Convert (x, y) into the 3D viewing space
		double M[16];
		glGetDoublev(GL_PROJECTION_MATRIX, M);
		//M is in column-major but inv_m is in row-major
		double inv_M[16];
		memset(inv_M, 0, sizeof(double)*16);
		inv_M[ 0]=1/M[0];
		inv_M[ 5]=1/M[5];
		inv_M[14]=1/M[14];
		inv_M[11]=-1;
		inv_M[15]=M[10]/M[14];
		double p0[4]={new_x, new_y, -1, 1}, p1[4];
		double q0[4]={new_x, new_y,  1, 1}, q1[4];
		Matrix_Vector_Product_4(inv_M, p0, p1);
		Matrix_Vector_Product_4(inv_M, q0, q1);
		
		// Convert (x ,y) into the 3D world space		
		glLoadIdentity();
		glTranslatef(center[0], center[1], center[2]);
		glRotated(-z_angle, 0, 0, -1);
		glRotatef(-swing_angle, 0, 1, 0);
		glRotatef(-elevate_angle, 1, 0, 0);
		glTranslatef(0, 0, zoom);
		glGetDoublev(GL_MODELVIEW_MATRIX, M);
		Matrix_Transpose_4(M, M);
		Matrix_Vector_Product_4(M, p1, p0);
		Matrix_Vector_Product_4(M, q1, q0);

		p[0]=p0[0]/p0[3];
		p[1]=p0[1]/p0[3];
		p[2]=p0[2]/p0[3];
		q[0]=q0[0]/q0[3];
		q[1]=q0[1]/q0[3];
		q[2]=q0[2]/q0[3];
	}


	static void Handle_SpecialKeypress(int key, int x, int y)
	{		
		if(key==100)		swing_angle+=3;
		else if(key==102)	swing_angle-=3;
		else if(key==103)	elevate_angle-=3;
		else if(key==101)	elevate_angle+=3;
		Handle_Reshape(screen_width, screen_height); 
		glutPostRedisplay();
	}

	 void Handle_Mouse_Move(int x, int y)
	{
		if(motion_mode!=NO_MOTION)
		{
			if(motion_mode==ROTATE_MOTION) 
			{
				swing_angle   += (double)(x - mouse_x)*360/(double)screen_width;
				elevate_angle += (double)(y - mouse_y)*360/(double)screen_height;

				z_angle+= atan2((double)(x - mouse_x), (double)(y - mouse_y))/4;
			//	z_angle += sqrt((double)(x - mouse_x)*(x - mouse_x) + (double)(y - mouse_y)*(y - mouse_y)) / 100;
		     //   if     (elevate_angle> 90)	elevate_angle = 90;
			//	else if(elevate_angle<-90)	elevate_angle = -90;

				if (z_angle> 90)	z_angle = 90;
				else if (z_angle<-90)	z_angle = -90;


			}
			if(motion_mode==ZOOM_MOTION)	zoom+=0.05 * (y-mouse_y);
			if(motion_mode==TRANSLATE_MOTION)
			{
				center[0] -= 0.1*(mouse_x - x);
				center[2] += 0.1*(mouse_y - y);
			}
			mouse_x=x;
			mouse_y=y;
			glutPostRedisplay();
		}


		
	}

	static void Handle_Mouse_Click(int button, int state, int x, int y)
	{
	}
	
};


//int		OPENGL_DRIVER::file_id			=	0;
//int		OPENGL_DRIVER::screen_width		=	1024;
//int		OPENGL_DRIVER::screen_height	=	768;
//int		OPENGL_DRIVER::mesh_mode		=	0;
//int		OPENGL_DRIVER::render_mode		=	0;
//double	OPENGL_DRIVER::zoom				=	30;
//double	OPENGL_DRIVER::swing_angle		=	-0;
//double	OPENGL_DRIVER::elevate_angle	=	0; 
//double	OPENGL_DRIVER::z_angle = -0;
//double	OPENGL_DRIVER::center[3]		=	{0, 0, 0};
//int		OPENGL_DRIVER::motion_mode		=	NO_MOTION;
//int		OPENGL_DRIVER::mouse_x			=	0;
//int		OPENGL_DRIVER::mouse_y			=	0;
//double** OPENGL_DRIVER::SphereCenter=0;
//double** OPENGL_DRIVER::joints = 0;
//int** OPENGL_DRIVER::joints_links = 0;
//double** OPENGL_DRIVER::SphereCenter_origin = 0;
MESH<FLOAT_TYPE>* OPENGL_DRIVER::mesh = 0;
//MESH<FLOAT_TYPE>* OPENGL_DRIVER::mesh_landmarks = 0;
//MESH<FLOAT_TYPE>* OPENGL_DRIVER::mesh_skeleton = 0;
//cv::Mat* OPENGL_DRIVER::TextureImage = NULL;
//GLubyte*** OPENGL_DRIVER::textBMP=NULL;
GLuint OPENGL_DRIVER::texName = 0;
//int OPENGL_DRIVER::geodesicDistance_mode=0;
vector<string> OPENGL_DRIVER::txtfileNames = {};
vector<Mat> OPENGL_DRIVER::textureImages = {};
vector<Mat> OPENGL_DRIVER::depthImages = {};
vector<Mat> OPENGL_DRIVER::maskImages = {};
vector<vector<Point2f>> OPENGL_DRIVER::joints_all = {};
#endif

