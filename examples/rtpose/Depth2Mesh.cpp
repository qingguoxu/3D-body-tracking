#include "Depth2Mesh.h"


Mat depthmap2pointmap(Mat& depth){
	Mat1i U_temp, V_temp;
	int height = depth.rows;
	int width = depth.cols;

	//clock_t begin = clock();






	meshgridTest(Range(0.5, width - 0.5), Range(0.5, height - 0.5), U_temp, V_temp);



	Mat U, V;
	Mat Z = depth;
	U_temp.convertTo(U, CV_32FC1);
	V_temp.convertTo(V, CV_32FC1);
	//cout << depth.type() << endl;
	//cout << U.type() << endl;
	Mat Z_U = Z.mul(U);
	Mat Z_V = Z.mul(V);
	Mat KK_t = KK.t();
	Mat output = Mat(Z.rows, Z.cols, CV_32FC3);
	Mat temp = Mat(1, 3, CV_32FC1);
	Mat KK_t_inv = KK_t.inv();

	float *input_KK_t_inv = (float*)(KK_t_inv.data);


	for (int i = 0; i < Z.rows; i++)
	{
		for (int j = 0; j < Z.cols; j++)
		{
			//temp.at<float>(0, 0) = Z_U.at<float>(i, j);
			// temp.at<float>(0, 1) = Z_V.at<float>(i, j);
			// temp.at<float>(0, 2) = Z.at<float>(i, j);
			// temp.val[0] = Z_U.at<float>(i, j);
			// temp.val[1] = Z_V.at<float>(i, j);
			// temp.val[2] = Z.at<float>(i, j);

			float temp_v1 = Z_U.at<float>(i, j);
			float temp_v2 = Z_V.at<float>(i, j);
			float temp_v3 = Z.at<float>(i, j);

			float a1 = temp_v1*input_KK_t_inv[0] + temp_v2*input_KK_t_inv[3] + temp_v3*input_KK_t_inv[6];
			float a2 = temp_v1*input_KK_t_inv[1] + temp_v2*input_KK_t_inv[4] + temp_v3*input_KK_t_inv[7];
			float a3 = temp_v1*input_KK_t_inv[2] + temp_v2*input_KK_t_inv[5] + temp_v3*input_KK_t_inv[8];

			// Mat result = temp*KK_t.inv();
			output.at<Vec3f>(i, j)[0] = a1;// result.at<Vec3f>(0, 0)[0];
			output.at<Vec3f>(i, j)[1] = a2;// result.at<Vec3f>(0, 0)[1];
			output.at<Vec3f>(i, j)[2] = a3;// result.at<Vec3f>(0, 0)[2];


		}
	}

	//	 clock_t end1 = clock();
	//	 double elapsed_secs = double(end1 - begin) / CLOCKS_PER_SEC;
	//	 cout << "time elapsed1: " << elapsed_secs << endl;

	// float *input_temp = (float*)(temp.data);
	/* float *input_Z_U = (float*)(Z_U.data);
	float *input_Z_V = (float*)(Z_V.data);
	float *input_Z= (float*)(Z.data);
	float *input_KK_t_inv = (float*)(KK_t_inv.data);

	float *input_output = (float*)(output.data);
	for (int i = 0; i < Z.cols; i++)
	{
	for (int j = 0; j < Z.rows; j++)
	{

	float temp_v1 = input_Z_U[Z.cols*j + i];
	float temp_v2 = input_Z_V[Z.cols*j + i];
	float temp_v3 = input_Z[Z.cols*j + i];

	//temp.at<float>(0, 0) = Z_U.at<float>(i, j);
	// temp.at<float>(0, 1) = Z_V.at<float>(i, j);
	// temp.at<float>(0, 2) = Z.at<float>(i, j);
	// temp.val[0] = Z_U.at<float>(i, j);
	// temp.val[1] = Z_V.at<float>(i, j);
	// temp.val[2] = Z.at<float>(i, j);
	Mat result = temp*KK_t.inv();

	float a1 = temp_v1*input_KK_t_inv[0] + temp_v2*input_KK_t_inv[3] + temp_v3*input_KK_t_inv[6];
	float a2 = temp_v1*input_KK_t_inv[1] + temp_v2*input_KK_t_inv[4] + temp_v3*input_KK_t_inv[7];
	float a3 = temp_v1*input_KK_t_inv[2] + temp_v2*input_KK_t_inv[5] + temp_v3*input_KK_t_inv[8];
	input_output[Z.cols*j + i + 0] = a1;
	input_output[Z.cols*j + i + 1] = a2;
	input_output[Z.cols*j + i + 2] = a3;

	// output.at<Vec3f>(i, j)[0] = result.at<Vec3f>(0, 0)[0];
	// output.at<Vec3f>(i, j)[1] = result.at<Vec3f>(0, 0)[1];
	// output.at<Vec3f>(i, j)[2] = result.at<Vec3f>(0, 0)[2];


	}
	}

	*/


	return output;

}



void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv,
	cv::Mat1i &X, cv::Mat1i &Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

void meshgridG(const cv::Mat &xgv, const cv::Mat &ygv,
	cv::Mat &X, cv::Mat &Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
	cv::Mat1i &X, cv::Mat1i &Y)
{
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
	meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}

static void meshgridTestG(const cv::Range &xgv, const cv::Range &ygv,
	cv::Mat &X, cv::Mat &Y)
{
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
	meshgridG(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}


Mat im2bw(Mat src, double grayThresh)
{
	cv::Mat dst;
	cv::threshold(src, dst, grayThresh, 255, CV_THRESH_BINARY);
	return dst;
}

Mat MaskBlob(Mat& mask){

	//keeps n biggest blobs in the mask
	Mat output = mask.clone();
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(output, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);



	for (int idx = 0; idx < contours.size(); idx++)
	{
		if (contours[idx].size() > 30){
			Scalar color(rand() & 255, rand() & 255, rand() & 255);
			drawContours(output, contours, idx, color, CV_FILLED, 8, hierarchy);
		}
	}



	return output;
}


void  selectmesh_vind(MESH<FLOAT_TYPE>& mesh, Mat& vind, int width, int height, MESH<FLOAT_TYPE>& submesh){

	//MESH<FLOAT_TYPE> submesh;
	int vnum = mesh.number;

	vector<bool>validv(vnum, false);


	for (int i = 0; i < vind.rows; i++)
	{
		// cout << vind.at<Point>(i).x << endl;
		// cout << vind.at<Point>(i).y << endl;
		int temp_ind = vind.at<Point>(i).x*height + vind.at<Point>(i).y;
		validv[temp_ind] = true;
		// cout << validv[temp_ind] << endl;
	}

	int subvnum = vind.rows;
	vector<int> allind(vnum, 0);

	for (int i = 0; i <subvnum; i++)
	{

		int temp_ind = vind.at<Point>(i).x*height + vind.at<Point>(i).y;
		allind[temp_ind] = i;

	}

	/////////////////////// step1 :determine if a triangle is valid

	vector<Vec3i> subi;

	for (int i = 0; i < mesh.t_number; i++)
	{
		if (validv[mesh.T[i * 3 + 0]] == true && validv[mesh.T[i * 3 + 1]] == true && validv[mesh.T[i * 3 + 2]] == true){

			Vec3i t;
			t[0] = mesh.T[i * 3 + 0];
			t[1] = mesh.T[i * 3 + 1];
			t[2] = mesh.T[i * 3 + 2];
			subi.push_back(t);

		}
	}
	// change the face index to new indices in vind


	for (int i = 0; i < subi.size(); i++)
	{
		subi[i][0] = allind[subi[i][0]];
		subi[i][1] = allind[subi[i][1]];
		subi[i][2] = allind[subi[i][2]];
	}



	/////////////////////// step2: determine if a vertex is referenced in the valid triangles

	//is vertex from step1 an unreference point?, valid is the same size of vind

	vector<bool> valid(subvnum, false);

	for (int i = 0; i < subi.size(); i++)
	{
		valid[subi[i][0]] = true;
		valid[subi[i][1]] = true;
		valid[subi[i][2]] = true;
	}

	///////////////////// compute the subv and its ind, ind2
	int start_id = 0;
	vector<int> ind;

	vector<int> allind2(subvnum, 0);
	for (int i = 0; i < valid.size(); i++)
	{
		if (valid[i] == true)
		{
			int temp_ind = vind.at<Point>(i).x*height + vind.at<Point>(i).y;
			ind.push_back(temp_ind);
			submesh.X[start_id * 3 + 0] = mesh.X[temp_ind * 3 + 0];
			submesh.X[start_id * 3 + 1] = mesh.X[temp_ind * 3 + 1];
			submesh.X[start_id * 3 + 2] = mesh.X[temp_ind * 3 + 2];

			submesh.VT[start_id * 2 + 0] = mesh.VT[temp_ind * 2 + 0];
			submesh.VT[start_id * 2 + 1] = mesh.VT[temp_ind * 2 + 1];

			allind2[i] = start_id;
			start_id++;

		}



	}
	submesh.number = start_id;
	///////////////////// recompute the face index to new indices in subv
	int subvnum_final = start_id;
	submesh.vt_number = start_id;

	submesh.t_number = subi.size();
	for (int i = 0; i < subi.size(); i++)
	{


		// Vec3i t;
		/// subi[i][0] = allind2[subi[i][0]];
		// subi[i][1] = allind2[subi[i][1]];
		// subi[i][2] = allind2[subi[i][2]];
		submesh.T[i * 3 + 0] = allind2[subi[i][0]];
		submesh.T[i * 3 + 1] = allind2[subi[i][1]];
		submesh.T[i * 3 + 2] = allind2[subi[i][2]];


	}

	///// recompute the texture coordinates, normals if needed
	/* int vt_num = mesh.vt_number;
	submesh.vt_number = mesh.vt_number;
	if (vt_num > 0)
	{
	for (int i = 0; i< vt_num; i++)
	{
	submesh.VT[i * 2 + 0] = mesh.VT[i * 2 + 0];
	submesh.VT[i * 2 + 1] = mesh.VT[i * 2 + 2];
	// cout << submesh.VT[i * 2 + 0] << ", " << submesh.VT[i * 2 + 1] << endl;
	}

	}*/

	// submesh.Write_OBJ("test2.obj");

	//return submesh;

}

void pointmap2mesh(Mat& pointmap, Mat& mask, MESH<FLOAT_TYPE>& submesh){

	MESH<FLOAT_TYPE>		mesh;
	int m = pointmap.rows;//height
	int n = pointmap.cols;//width

	Mat channel[3];
	split(pointmap, channel);

	Mat channel_t[3];

	channel_t[0] = channel[0].t();
	channel_t[1] = channel[1].t();
	channel_t[2] = channel[2].t();

	Mat pointmap_permute;

	merge(channel_t, 3, pointmap_permute);


	Mat v = pointmap_permute.reshape(1, m*n);


	Mat umap, vmap;

	meshgridTestG(Range(0.5, n - 0.5), Range(0.5, m - 0.5), umap, vmap);

	Mat channel_uv[2];

	channel_uv[0] = umap.t();
	channel_uv[1] = vmap.t();
	Mat temp;
	merge(channel_uv, 2, temp);


	// cout << temp.type() << endl;
	Mat uv = temp.reshape(1, m*n);
	// cout << uv.type() << endl;
	Mat vt = uv.clone();

	vt.convertTo(vt, CV_32FC2);



	for (int i = 0; i < vt.rows; i++)
	{


		vt.at<float>(i, 0) = vt.at<float>(i, 0) / n;
		vt.at<float>(i, 1) = 1 - vt.at<float>(i, 1) / m;


	}

	// generate faces

	Mat f = Mat::zeros((m - 1)*(n - 1) * 2, 3, CV_32F);


	for (int i = 1; i <= n - 1; i++)
	{
		for (int j = 1; j <= m - 1; j++)
		{
			int t1 = (i - 1)*m + j - 1;
			int t2 = t1 + m;


			f.at<float>((i - 1)*(m - 1) * 2 + (j - 1) * 2, 0) = t2;
			f.at<float>((i - 1)*(m - 1) * 2 + (j - 1) * 2 + 1, 0) = t2 + 1;

			f.at<float>((i - 1)*(m - 1) * 2 + (j - 1) * 2, 1) = t1;
			f.at<float>((i - 1)*(m - 1) * 2 + (j - 1) * 2 + 1, 1) = t1;

			f.at<float>((i - 1)*(m - 1) * 2 + (j - 1) * 2, 2) = t2 + 1;
			f.at<float>((i - 1)*(m - 1) * 2 + (j - 1) * 2 + 1, 2) = t1 + 1;




		}
	}

	// fill mesh

	mesh.number = v.rows;
	mesh.vt_number = v.rows;
	mesh.t_number = f.rows;
	for (int i = 0; i < mesh.number; i++)
	{
		mesh.X[i * 3 + 0] = v.at<float>(i, 0);
		mesh.X[i * 3 + 1] = v.at<float>(i, 1);
		mesh.X[i * 3 + 2] = v.at<float>(i, 2);



		mesh.VT[i * 2 + 0] = vt.at<float>(i, 0);
		mesh.VT[i * 2 + 1] = vt.at<float>(i, 1);

		// if (mesh.VT[i * 2 + 0] != 0){
		// cout << vt.type() << endl;
		//	cout << mesh.VT[i * 2 + 0] << ", " << mesh.VT[i * 2 + 1] << endl;
		// }

	}


	for (int i = 0; i < mesh.t_number; i++)
	{
		mesh.T[i * 3 + 0] = f.at<float>(i, 0);
		mesh.T[i * 3 + 1] = f.at<float>(i, 1);
		mesh.T[i * 3 + 2] = f.at<float>(i, 2);




	}



	// extract the non zeros pixels in mask

	Mat locations;
	findNonZero(mask, locations);
	// MESH<FLOAT_TYPE> submesh;
	if (locations.rows != v.rows){


		selectmesh_vind(mesh, locations, n, m, submesh);


	}



}


void RemoveLongFace(MESH<FLOAT_TYPE>& mesh, float threshold, MESH<FLOAT_TYPE>& mesh_clean){

	int face_counter = 0;
	for (int i = 0; i < mesh.t_number; i++)
	{
		int ind_v[3];
		ind_v[0] = mesh.T[i * 3 + 0];
		ind_v[1] = mesh.T[i * 3 + 1];
		ind_v[2] = mesh.T[i * 3 + 2];

		Point3f v0(mesh.X[ind_v[0] * 3 + 0], mesh.X[ind_v[0] * 3 + 1], mesh.X[ind_v[0] * 3 + 2]);
		Point3f v1(mesh.X[ind_v[1] * 3 + 0], mesh.X[ind_v[1] * 3 + 1], mesh.X[ind_v[1] * 3 + 2]);
		Point3f v2(mesh.X[ind_v[2] * 3 + 0], mesh.X[ind_v[2] * 3 + 1], mesh.X[ind_v[2] * 3 + 2]);

		Point3f edge1(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
		Point3f edge2(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
		Point3f edge3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);

		vector<float> edge_length(3);
		edge_length[0] = sqrt(edge1.x*edge1.x + edge1.y*edge1.y + edge1.z*edge1.z);
		edge_length[1] = sqrt(edge2.x*edge2.x + edge2.y*edge2.y + edge2.z*edge2.z);
		edge_length[2] = sqrt(edge3.x*edge3.x + edge3.y*edge3.y + edge3.z*edge3.z);

		auto smallest = std::min_element(std::begin(edge_length), std::end(edge_length));

		auto biggest = std::max_element(std::begin(edge_length), std::end(edge_length));

		if (*smallest > 0 && *biggest < threshold)
		{
			mesh_clean.T[face_counter * 3 + 0] = mesh.T[i * 3 + 0];
			mesh_clean.T[face_counter * 3 + 1] = mesh.T[i * 3 + 1];
			mesh_clean.T[face_counter * 3 + 2] = mesh.T[i * 3 + 2];
			face_counter++;
		}



	}

	mesh_clean.t_number = face_counter;
	mesh_clean.number = mesh.number;
	mesh_clean.vt_number = mesh.vt_number;

	for (int i = 0; i < mesh.number; i++)
	{
		mesh_clean.X[i * 3 + 0] = mesh.X[i * 3 + 0];
		mesh_clean.X[i * 3 + 1] = mesh.X[i * 3 + 1];
		mesh_clean.X[i * 3 + 2] = mesh.X[i * 3 + 2];



		mesh_clean.VT[i * 2 + 0] = mesh.VT[i * 2 + 0];
		mesh_clean.VT[i * 2 + 1] = mesh.VT[i * 2 + 1];

		// cout << mesh_clean.VT[i * 2 + 0] << ", " << mesh_clean.VT[i * 2 + 1] << endl;
	}


	// mesh.Write_OBJ("test6.obj");
	// mesh_clean.Write_OBJ("test7.obj");

}