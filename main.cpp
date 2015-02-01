/*
Descriptor Matching Analyzer

Copyright (c) 2015 Gou Koutaki

This software is released under the BSD License.

*/


#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <fstream>

using namespace cv;

#define max(a,b) (a<b?b:a)
#define min(a,b) (a>b?b:a)

struct SIdx
{
    SIdx() : S(-1), i1(-1), i2(-1) {}
    SIdx(float _S, int _i1, int _i2) : S(_S), i1(_i1), i2(_i2) {}
    float S;
    int i1;
    int i2;

    bool operator<(const SIdx& v) const { return S > v.S; }

    struct UsedFinder
    {
        UsedFinder(const SIdx& _used) : used(_used) {}
        const SIdx& used;
        bool operator()(const SIdx& v) const { return  (v.i1 == used.i1 || v.i2 == used.i2); }
        UsedFinder& operator=(const UsedFinder&);
    };
};


void evaluateFeatureDetector2( const Mat& img1, const Mat& img2, const Mat& H1to2,
                              vector<KeyPoint>* _keypoints1, vector<KeyPoint>* _keypoints2,
                              float& repeatability, int& correspCount, vector<SIdx> & overlaps);

//#include <direct.h>
void gen_dir(char *file){
	//_mkdir(file);
}



struct ROC_gr {
	Mat img;
	int w, h;
	int sw, sh;
	int x0, y0;
	int bw, bh;

	void puttext_v(Mat & img, char *txt, Point pos, Scalar col){
		for(int i = 0;i < strlen(txt);i++){
			char buf[256];
			sprintf(buf, "%c", txt[i]);
			putText(img, buf, pos + Point(0, i * 20), FONT_HERSHEY_PLAIN, 0.9, col);
		}
	};

	Point get_pos(float x, float y){

		int rx, ry;

		rx = x0 + x * sw;
		ry = y0 + (1-y) * sh;

		return Point(rx,ry);
	};
	void init_graph(){

		img = Mat(h, w, CV_8UC3);
		rectangle(img, Point(0,0), Point(w,h), Scalar(255,255,255), -1);

		for(int i = 0;i <= 10.0;i++){
			line(img, Point(x0 + i * bw, y0), Point(x0 + i * bw, y0 + sh), Scalar(0,0,0), 1);
			line(img, Point(x0, y0 + i * bh), Point(x0 + sw, y0 + i * bh), Scalar(0,0,0), 1);
			char txt[256];
			sprintf(txt, "%.1f", i / 10.0);
			putText(img, txt, Point(x0 + i * bw - 10, y0 + sh + 15), FONT_HERSHEY_PLAIN , 0.9, Scalar(0,0,0));
			putText(img, txt, Point(x0 - 30, y0 + sh - i * bh), FONT_HERSHEY_PLAIN , 0.9, Scalar(0,0,0));
		}
		putText(img, "1 - precision", Point(x0 + sw * 0.5 - 40, y0 + sh + 50), FONT_HERSHEY_PLAIN , 0.9, Scalar(0,0,0));
		puttext_v(img, "recall", Point(20, 80), Scalar(0,0,0));
	}

	void draw(float *x, float *y, int N){

		for(int i = 0;i < N - 1;i++){
			line(img, get_pos(x[i],y[i]), get_pos(x[i+1],y[i+1]), Scalar(0,255,0));
		}
		for(int i = 0;i < N;i++){
			putText(img, "x", get_pos(x[i],y[i])+Point(-5,4), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
		}
		//imshow("ROC", img);
		//cv::waitKey(0);
	};

	ROC_gr(){
		w = 480;
		h = 370;
		sw = 360;
		sh = 260;
		x0 = 80;
		y0 = 50;
		bw = sw / 10;
		bh = sh / 10;
	};
};

void readtxt(char *filename, Mat & H){

	H = Mat_<float>(3,3);

	char line[1024];
	FILE *fp=fopen(filename, "r");
	fgets(line, 1024, fp);
	sscanf(line,"%f %f %f",  &H.at<float>(0,0),  &H.at<float>(0,1),  &H.at<float>(0,2));
	fgets(line, 1024, fp);
	sscanf(line,"%f %f %f",  &H.at<float>(1,0),  &H.at<float>(1,1),  &H.at<float>(1,2));
	fgets(line, 1024, fp);
	sscanf(line,"%f %f %f",  &H.at<float>(2,0),  &H.at<float>(2,1),  &H.at<float>(2,2));
	fclose(fp);

}

int round(float x){

	return (int)floor(x+0.5);
}




void read_kpt(char *filename, std::vector<KeyPoint> & keys){

	FILE *fp=fopen(filename, "r");

	if(fp){
		int N;
		char buf[1024];
		fgets(buf, 1024, fp);
		sscanf(buf, "%d", &N);
		keys.resize(N);
		for(int i = 0;i < N;i++){
			fgets(buf, 1024, fp);
			float x, y, a, b, c, scale;
			sscanf(buf, "%f %f %f %f %f %f", &x,&y,&a,&b,&c, &scale);
			scale = scale / 3.0;
			keys[i].pt.x = x;
			keys[i].pt.y = y;
			keys[i].size = scale;
		}
		fclose(fp);
	}
}

void draw_descriptors(Mat & img, float *d1, float *d2, Point & pt){

	float max_v = 0;
	for(int i = 0;i < 128;i++){
		max_v = max(max_v, fabs(d1[i]));
		max_v = max(max_v, fabs(d2[i]));
	}

	rectangle(img, pt, pt + Point(128, 64), Scalar(255,255,255), -1);
	for(int i = 0;i < 128;i++){
		line(img, pt + Point(i, 16 - 16 * d1[i]/max_v), pt + Point(i, 16), Scalar(64,64,64));
		line(img, pt + Point(i, 48 - 16 * d2[i]/max_v), pt + Point(i, 48), Scalar(64,64,64));
	}

}

Point2f calc_project(Point2f & in, Mat & H){

	float rx,ry,rz;
	rx = in.x * H.at<float>(0,0) + in.y * H.at<float>(0,1) + H.at<float>(0,2);
	ry = in.x * H.at<float>(1,0) + in.y * H.at<float>(1,1) + H.at<float>(1,2);
	rz = in.x * H.at<float>(2,0) + in.y * H.at<float>(2,1) + H.at<float>(2,2);

	return Point2f(rx/rz, ry/rz);
};

int check(Point pt, int w, int h){
	if(pt.x < 32 || pt.x > (w-32) || pt.y < 32 || pt.y > (h-32))
		return 0;
	return 1;
};


FILE *rp;

int main(int argc, char **argv){
	

	ROC_gr roc;
	roc.init_graph();


	Mat img1,img2, H;
	img1 = imread(argv[4]);
	img2 = imread(argv[5]);
	
	readtxt(argv[3], H);

#ifdef USE_SURF
	SURF sift;
#else
	SIFT sift;
#endif

	std::vector<KeyPoint> keypoints1;
	std::vector<KeyPoint> keypoints2;
	Mat descriptor1, descriptor2;

	// read
	if(argc==6){
		printf("key1=%s, key2=%s " , argv[1], argv[2]);
		cv::FileStorage fs1(argv[1], cv::FileStorage::READ);
		cv::FileNode kptFileNode1 = fs1["keypoints_1"];
		read( kptFileNode1, keypoints1 );
		kptFileNode1 = fs1["descriptors_1"];
		read( kptFileNode1, descriptor1 );
		fs1.release();	

		cv::FileStorage fs2(argv[2], cv::FileStorage::READ);
		cv::FileNode kptFileNode2 = fs2["keypoints_1"];
		read( kptFileNode2, keypoints2 );
		kptFileNode2 = fs2["descriptors_1"];
		read( kptFileNode2, descriptor2 );
		fs2.release();	
	} else {
		read_kpt(argv[1], keypoints1);
		read_kpt(argv[2], keypoints2);
	}

	char report_dir[256];
	char report_file[256];

	sprintf(report_dir, "./");
	sprintf(report_file, "%s/index.html", report_dir);

	gen_dir(report_dir);


	rp = fopen(report_file,"w");

	fprintf(rp, "<HTML><BODY><H1>Descriptor matching</H1>");

	keypoints1.resize(min(keypoints1.size(), 2000));
	keypoints2.resize(min(keypoints2.size(), 2000));
	for(int i = 0;i < keypoints1.size();i++){
		keypoints1[i].class_id = i;
	}
	for(int j = 0;j < keypoints2.size();j++){
		keypoints2[j].class_id = j;
	}

	float repeatablity;
	int correspCount;

	vector<SIdx> overlaps;
	evaluateFeatureDetector2(img1, img2, H, &keypoints1, &keypoints2, repeatablity, correspCount, overlaps);

	printf("rep.=%f\n", repeatablity);

	Mat ndescriptor1 = descriptor1, ndescriptor2 = descriptor2;
	for(int i = 0;i < keypoints1.size();i++){
		int index = (int)keypoints1[i].class_id;
		for(int k = 0;k < 128;k++){
			ndescriptor1.at<float>(i,k)=descriptor1.at<float>(index,k);
		}
	}
	for(int i = 0;i < keypoints2.size();i++){
		int index = (int)keypoints2[i].class_id;
		for(int k = 0;k < 128;k++){
			ndescriptor2.at<float>(i,k)=descriptor2.at<float>(index,k);
		}
	}
	descriptor1 = ndescriptor1;


	vector<SIdx> detect_overlaps;
	vector<float> ratiolist;
	for(int i = 0;i < keypoints1.size();i++){
		int min_j = 0;
		float min_L = 9999999, second_min = 99999;
		for(int j = 0;j < keypoints2.size();j++){
			float L = 0;
			for(int k = 0;k < 128;k++){
				float v1 = descriptor1.at<float>(i, k);
				float v2 = descriptor2.at<float>(j, k);
				float v = v1 - v2;
				//printf("%d,", (int)v2);
				L = L + v * v;
			}
			L = sqrtf(L/128.0);
			if(L < min_L){
				second_min = min_L;
				min_L = L;
				min_j = j;
			} else if(L < second_min){
				second_min = L;
			}
		}
		float ratio = min_L / second_min;
		ratiolist.push_back(ratio);
		SIdx t;
		t.i1 = i;
		t.i2 = min_j;
		t.S  = ratio;
		detect_overlaps.push_back(t);
	}

	int ctbl[10000];
	for(int i = 0;i < detect_overlaps.size();i++){
		ctbl[i] = 0;
		for(int j = 0;j < overlaps.size();j++){
			if(detect_overlaps[i].i1 == overlaps[j].i1 && detect_overlaps[i].i2 == overlaps[j].i2){
				ctbl[i] = 1;
				goto ENDL;
			}
		}
ENDL:
		;
	}

	printf("\n");
	FILE *fp=fopen("./pr.txt","w");

	float tbl_precision[256], tbl_recall[256];
	int   tbl_n = 0;
	for(float thresh = 0.20;thresh <= 1.05;thresh += 0.05){
		int correct_n = 0;
		int detect_num = 0;
		int false_n = 0;

		for(int i = 0;i < ratiolist.size();i++){
			float ratio = ratiolist[i];
			if(ratio < thresh){	
				detect_num++;
				if(ctbl[i] == 1)
					correct_n++;
				else 
					false_n++;

			}
		}

		float recall, precision;

		recall    = correct_n / (float)overlaps.size();
		precision = false_n / (float)(0.000001+correct_n + false_n);;

		printf("T=%.2f, cn=%d, dn=%d, ref=%d, recall=%.3f, 1-precision=%.3f\n", thresh, correct_n,  detect_num, overlaps.size(), recall, precision);
		fprintf(fp, "%f, %f\n", precision, recall);

		tbl_precision[tbl_n] = precision;
		tbl_recall[tbl_n++]  = recall;
	}
	fclose(fp);


	vector<SIdx> detect_overlaps2;
	for(int i = 0;i < detect_overlaps.size();i++){
		if(detect_overlaps[i].S < 0.70){
			detect_overlaps2.push_back(detect_overlaps[i]);
		}
	}
	detect_overlaps = detect_overlaps2;
	int correct_n = 0;
	int detect_num = 0;
	int false_n = 0;

	for(int i = 0;i < detect_overlaps.size();i++){
		ctbl[i] = 0;
		detect_num++;
		for(int j = 0;j < overlaps.size();j++){
			if(detect_overlaps[i].i1 == overlaps[j].i1 && detect_overlaps[i].i2 == overlaps[j].i2){
				ctbl[i] = 1;
				correct_n++;
				goto ENDL2;
			}
		}
		false_n++;
		for(int j = 0;j < overlaps.size();j++){
			if(detect_overlaps[i].i1 == overlaps[j].i1){
				ctbl[i] |= 2;
			}
			if(detect_overlaps[i].i2 == overlaps[j].i2){
				ctbl[i] |= 4;
			}
		}

ENDL2:
		;
	}
	float recall    = correct_n / (float)overlaps.size();
	float precision = false_n / (float)(correct_n + false_n);;



	int w1 = img1.cols, h1 = img1.rows;
	int w2 = img2.cols, h2 = img2.rows;
	Mat outimg(Size(w1+w2, max(h1,h2)), CV_8UC3);
	Mat roi1(outimg, Rect(0,  0, w1, h1));
	Mat roi2(outimg, Rect(w1, 0, w2, h2));
	img1.copyTo(roi1);
	img2.copyTo(roi2);

	int reason_n = 0;

	for(int i = 0;i < detect_overlaps.size();i++){
		int i1, i2;
		i1 = detect_overlaps[i].i1;
		i2 = detect_overlaps[i].i2;

		Point rpt1 = Point(keypoints1[i1].pt.x, keypoints1[i1].pt.y), rpt2(keypoints2[i2].pt.x, keypoints2[i2].pt.y);

		float s = 1;
		if(check(rpt1, w1, h1) == 0 || check(rpt2, w2, h2) == 0)
			s = 0.2;

		if(ctbl[i]==1){
			circle(outimg, keypoints1[i1].pt, 3, Scalar(255,0,0), 1, CV_AA);
			circle(outimg, keypoints2[i2].pt+ Point2f(w1, 0), 3, Scalar(255,0,0), 1, CV_AA);
		}
		else if(ctbl[i]==0){
			circle(outimg, keypoints1[i1].pt, 3, Scalar(0,0,255*s), 1, CV_AA);
			circle(outimg, keypoints2[i2].pt+ Point2f(w1, 0), 3, Scalar(0,0,255*s), 1, CV_AA);
		} else {
			if(ctbl[i]==2){
			circle(outimg, keypoints1[i1].pt, 3, Scalar(0,255*s,255*s), 1, CV_AA);
			}
			if(ctbl[i]==4){
			circle(outimg, keypoints2[i2].pt+ Point2f(w1, 0), 3, Scalar(0,255*s,255*s), 1, CV_AA);
			}
		}
	}

	char outfile[256];
	sprintf(outfile, "%s/result.png", report_dir);
	imwrite(outfile, outimg);
	fprintf(rp, "<br><H2>Matching result: ratio test with T=0.7</H2><img src=\"result.png\" usemap=\"#top\">\n<map name=\"top\">\n");

	int link_n = 0;
	for(int i = 0;i < detect_overlaps.size();i++){
		int i1, i2;
		i1 = detect_overlaps[i].i1;
		i2 = detect_overlaps[i].i2;
		if(ctbl[i]!=1){

			Point rpt1 = Point(keypoints1[i1].pt.x, keypoints1[i1].pt.y), rpt2(keypoints2[i2].pt.x, keypoints2[i2].pt.y);

			if(check(rpt1, w1, h1) == 0)
				continue;
			if(check(rpt2, w2, h2) == 0)
				continue;

			Point pt1 = keypoints1[i1].pt;
			Point pt2 = keypoints2[i2].pt + Point2f(w1, 0);
			fprintf(rp, "<area shape=\"circle\" coords=\"%d,%d,%d\" href=\"./index.html#A%d\">\n", pt1.x, pt1.y, 5, link_n);
			fprintf(rp, "<area shape=\"circle\" coords=\"%d,%d,%d\" href=\"./index.html#A%d\">\n", pt2.x, pt2.y, 5, link_n);
			link_n++;
		}
	}
	fprintf(rp, "</map>\n");


	fprintf(rp, "<H2 bgcolor=black><font color=blue>Blue: Correct  </font><font color=red>Red: False   </font><font color=orange>Yellow: Mismatch  </font><br><br></H2><H2>Left:%d keypoints, Right:%d keypoints<br>\nReference correspondings = %d<br>\nDetected correspondings = %d<br>\nCorrect number =%d<br>\n<br>Recall=%.2f, 1-Precision=%.2f<br></H2>", keypoints1.size(), keypoints1.size(), overlaps.size(), detect_overlaps.size(), correct_n, recall , precision);

	// gen ROC curve
	roc.draw(tbl_precision,tbl_recall,tbl_n);
	sprintf(outfile, "%s/roc.png", report_dir);
	imwrite(outfile, roc.img);
	fprintf(rp, "<br><hr><H1>ROC curve</H1><img src=\"roc.png\"><br><hr>\n");

	//********************************************************************************************************************
	// analyze false reason
	fprintf(rp, "<H2>False reason</H2><br>\n<table border=1>\n");
	for(int i = 0;i < detect_overlaps.size();i++){
		int i1, i2;
		i1 = detect_overlaps[i].i1;
		i2 = detect_overlaps[i].i2;
		if(ctbl[i]!=1){
			Mat reason(Size(256, 64), CV_8UC3);
			Point2f prj_pt;

			prj_pt = calc_project(keypoints1[i1].pt, H);

			float distance = norm(prj_pt - keypoints2[i2].pt);
			Point rpt1 = Point(keypoints1[i1].pt.x, keypoints1[i1].pt.y), rpt2(keypoints2[i2].pt.x, keypoints2[i2].pt.y);

			if(check(rpt1, w1, h1) == 0)
				continue;
			if(check(rpt2, w2, h2) == 0)
				continue;



			Mat roi1s(img1,   Rect(rpt1 + Point(-32,-32), rpt1 + Point(32,32)));
			Mat roi1d(reason, Rect(0, 0, 64, 64));
			Mat roi2s(img2,   Rect(rpt2 + Point(-32,-32), rpt2 + Point(32,32)));
			Mat roi2d(reason, Rect(64, 0, 64, 64));
			roi1s.copyTo(roi1d);
			roi2s.copyTo(roi2d);


			int r1 = round(keypoints1[i1].size / 2);
			int r2 = round(keypoints2[i2].size / 2);
			float dir1 = 3.14159*keypoints1[i1].angle/180.0;
			float dir2 = 3.14159*keypoints2[i2].angle/180.0;



			circle(reason, Point(32, 32), 5, Scalar(0,0,255), 1.5, CV_AA);
			line(reason, Point(32, 32),  Point(32, 32) + Point(cos(dir1)*10,sin(dir1)*10), Scalar(0,0,255), 1.5, CV_AA);
			circle(reason, Point(64 + 32, 32), 5, Scalar(0,0,255), 1.5, CV_AA);
			line(reason, Point(64 + 32, 32),  Point(64 + 32, 32) + Point(cos(dir2)*10,sin(dir2)*10), Scalar(0,0,255), 1.5, CV_AA);




			for(int i = 0;i < overlaps.size();i++){
				if(overlaps[i].i1 == i1){
					Point ck = keypoints2[overlaps[i].i2].pt - keypoints2[i2].pt;
					if(norm(ck)<5){
					int   cr = round(keypoints2[overlaps[i].i2].size / 2);
					circle(reason, Point(64 + 32, 32) + ck, 5, Scalar(0,255,0), 1.5, CV_AA);
					}
				}
				if(overlaps[i].i2 == i2){
					Point ck = keypoints1[overlaps[i].i1].pt - keypoints1[i1].pt;
					if(norm(ck)<5){
					int   cr = round(keypoints1[overlaps[i].i1].size / 2);
					circle(reason, Point(32, 32) + ck, 5, Scalar(0,255,0), 1.5, CV_AA);
					}
				}
			}

			Point dpt(128, 0);
			draw_descriptors(reason,  descriptor1.ptr<float>((int)i1), descriptor2.ptr<float>((int)i2), dpt);
			//********************************************************************************************************************
			float desc_d = detect_overlaps[i].S;
			printf("center distance = %f\n", distance);
			printf("descrptor distance = %f\n", desc_d);

			char txt[256];
			sprintf(txt, "%s/img%03d.png", report_dir, reason_n);
			imwrite(txt, reason);
			sprintf(txt, "img%03d.png", reason_n);

			fprintf(rp, "<tr>");

			fprintf(rp, "<td><a name=\"A%d\">-</td>", reason_n);
			
			if(ctbl[i]==0){
				fprintf(rp, "<td bgcolor=red>");
				fprintf(rp, "#%d", reason_n);
				fprintf(rp, "</td><td>");
				fprintf(rp, "Refference does'nt contain the corresponding");
			} else if(ctbl[i]==2){
				fprintf(rp, "<td bgcolor=yellow>");
				fprintf(rp, "#%d", reason_n);
				fprintf(rp, "</td><td>");
				fprintf(rp, "Kpt1 matched another Kpt", reason_n);
				fprintf(rp, "</td>");
			} else if(ctbl[i]==4){
				fprintf(rp, "<td bgcolor=orange>");
				fprintf(rp, "#%d", reason_n);
				fprintf(rp, "</td><td>");
				fprintf(rp, "Kpt2 matched another Kpt", reason_n);
				fprintf(rp, "</td>");
			} else if(ctbl[i]==6){
				fprintf(rp, "<td bgcolor=Brown>");
				fprintf(rp, "#%d", reason_n);
				fprintf(rp, "</td><td>");
				fprintf(rp, "", reason_n+1);
				fprintf(rp, "</td>");
			}

			fprintf(rp, "<td>");
			fprintf(rp, "<img src=\"%s\"><br>", txt);
			fprintf(rp,"center distance = %f<br>\n", distance);
			fprintf(rp,"descrptor distance = %f<br>\n", desc_d);
			fprintf(rp, "</td>");

			
			if(distance < 3.0 &&  desc_d < 1.0){
				fprintf(rp, "<td bgcolor=grey>");
				fprintf(rp, "Maybe correct, but Ref. matched other neighbor Kpt</td>");
			}
			
			if(distance > 10.0  &&  desc_d < 1.0){
				fprintf(rp, "<td bgcolor=yellow>");
				fprintf(rp, "Mistake matching with similar descriptor</td>");
			}

			fprintf(rp, "</tr>\n");

			reason_n++;
			

		}
	}

	fprintf(rp, "</table>");

	fprintf(rp, "</HTML></BODY>");
	fclose(rp);

	return 0;
}

