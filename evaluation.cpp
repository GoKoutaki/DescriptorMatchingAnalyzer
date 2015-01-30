//*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <fstream>
#include <limits>


#define M_PI 3.14159

#undef rad1
#undef rad2


using namespace cv;
using namespace std;

template<typename _Tp> static int solveQuadratic(_Tp a, _Tp b, _Tp c, _Tp& x1, _Tp& x2)
{
    if( a == 0 )
    {
        if( b == 0 )
        {
            x1 = x2 = 0;
            return c == 0;
        }
        x1 = x2 = -c/b;
        return 1;
    }

    _Tp d = b*b - 4*a*c;
    if( d < 0 )
    {
        x1 = x2 = 0;
        return 0;
    }
    if( d > 0 )
    {
        d = std::sqrt(d);
        double s = 1/(2*a);
        x1 = (-b - d)*s;
        x2 = (-b + d)*s;
        if( x1 > x2 )
            std::swap(x1, x2);
        return 2;
    }
    x1 = x2 = -b/(2*a);
    return 1;
}

//for android ndk
#undef _S
static inline Point2f applyHomography( const Mat_<double>& H, const Point2f& pt )
{
    double z = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2);
    if( z )
    {
        double w = 1./z;
        return Point2f( (float)((H(0,0)*pt.x + H(0,1)*pt.y + H(0,2))*w), (float)((H(1,0)*pt.x + H(1,1)*pt.y + H(1,2))*w) );
    }
    return Point2f(0, 0 );
}

static inline void linearizeHomographyAt( const Mat_<double>& H, const Point2f& pt, Mat_<double>& A )
{
    A.create(2,2);
    double p1 = H(0,0)*pt.x + H(0,1)*pt.y + H(0,2),
           p2 = H(1,0)*pt.x + H(1,1)*pt.y + H(1,2),
           p3 = H(2,0)*pt.x + H(2,1)*pt.y + H(2,2),
           p3_2 = p3*p3;
    if( p3 )
    {
        A(0,0) = H(0,0)/p3 - p1*H(2,0)/p3_2; // fxdx
        A(0,1) = H(0,1)/p3 - p1*H(2,1)/p3_2; // fxdy

        A(1,0) = H(1,0)/p3 - p2*H(2,0)/p3_2; // fydx
        A(1,1) = H(1,1)/p3 - p2*H(2,1)/p3_2; // fydx
    }
    else
        A.setTo(0);
}

class EllipticKeyPoint
{
public:
    EllipticKeyPoint();
    EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse );

    static void convert( const vector<KeyPoint>& src, vector<EllipticKeyPoint>& dst );
    static void convert( const vector<EllipticKeyPoint>& src, vector<KeyPoint>& dst );

    static Mat_<double> getSecondMomentsMatrix( const Scalar& _ellipse );
    Mat_<double> getSecondMomentsMatrix() const;

    void calcProjection( const Mat_<double>& H, EllipticKeyPoint& projection ) const;
    static void calcProjection( const vector<EllipticKeyPoint>& src, const Mat_<double>& H, vector<EllipticKeyPoint>& dst );

	int class_id;
	float angle;
    Point2f center;
    Scalar ellipse; // 3 elements a, b, c: ax^2+2bxy+cy^2=1
    Size_<float> axes; // half lenght of elipse axes
    Size_<float> boundingBox; // half sizes of bounding box which sides are parallel to the coordinate axes
};

EllipticKeyPoint::EllipticKeyPoint()
{
    *this = EllipticKeyPoint(Point2f(0,0), Scalar(1, 0, 1) );
}

EllipticKeyPoint::EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse )
{
    center = _center;
    ellipse = _ellipse;

    double a = ellipse[0], b = ellipse[1], c = ellipse[2];
    double ac_b2 = a*c - b*b;
    double x1, x2;
    solveQuadratic(1., -(a+c), ac_b2, x1, x2);
    axes.width = (float)(1/sqrt(x1));
    axes.height = (float)(1/sqrt(x2));

    boundingBox.width = (float)sqrt(ellipse[2]/ac_b2);
    boundingBox.height = (float)sqrt(ellipse[0]/ac_b2);
}

Mat_<double> EllipticKeyPoint::getSecondMomentsMatrix( const Scalar& _ellipse )
{
    Mat_<double> M(2, 2);
    M(0,0) = _ellipse[0];
    M(1,0) = M(0,1) = _ellipse[1];
    M(1,1) = _ellipse[2];
    return M;
}

Mat_<double> EllipticKeyPoint::getSecondMomentsMatrix() const
{
    return getSecondMomentsMatrix(ellipse);
}

void EllipticKeyPoint::calcProjection( const Mat_<double>& H, EllipticKeyPoint& projection ) const
{
    Point2f dstCenter = applyHomography(H, center);

    Mat_<double> invM; invert(getSecondMomentsMatrix(), invM);
    Mat_<double> Aff; linearizeHomographyAt(H, center, Aff);
    Mat_<double> dstM; invert(Aff*invM*Aff.t(), dstM);

	float x = center.x;
	float y = center.y;
	float x2 = Aff(0,0)*x+Aff(0,1)*y;
	float y2 = Aff(1,0)*x+Aff(1,1)*y;

	float x3 = H(0,0)*x+H(0,1)*y + H(0,2);
	float y3 = H(1,0)*x+H(1,1)*y + H(1,2);
	float z3 = H(2,0)*x+H(2,1)*y + H(2,2);

	float dx = x3 / z3;
	float dy = y3 / z3;

	float x0 = H(0,2) / H(2,2);
	float y0 = H(1,2) / H(2,2);

    projection = EllipticKeyPoint( dstCenter, Scalar(dstM(0,0), dstM(0,1), dstM(1,1)) );
}

void EllipticKeyPoint::convert( const vector<KeyPoint>& src, vector<EllipticKeyPoint>& dst )
{
    if( !src.empty() )
    {
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            float rad = src[i].size/2;
            assert( rad );
            float fac = 1.f/(rad*rad);
            dst[i] = EllipticKeyPoint( src[i].pt, Scalar(fac, 0, fac) );
			dst[i].angle = src[i].angle;
			dst[i].class_id = src[i].class_id;
        }
    }
}

void EllipticKeyPoint::convert( const vector<EllipticKeyPoint>& src, vector<KeyPoint>& dst )
{
    if( !src.empty() )
    {
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            Size_<float> axes = src[i].axes;
            float rad = sqrt(axes.height*axes.width);
            dst[i] = KeyPoint(src[i].center, 2*rad );
			dst[i].angle = src[i].angle;
			dst[i].class_id = src[i].class_id;
        }
    }
}

void EllipticKeyPoint::calcProjection( const vector<EllipticKeyPoint>& src, const Mat_<double>& H, vector<EllipticKeyPoint>& dst )
{
    if( !src.empty() )
    {
        assert( !H.empty() && H.cols == 3 && H.rows == 3);
        dst.resize(src.size());
        vector<EllipticKeyPoint>::const_iterator srcIt = src.begin();
        vector<EllipticKeyPoint>::iterator       dstIt = dst.begin();
        for( ; srcIt != src.end(); ++srcIt, ++dstIt ){
            srcIt->calcProjection(H, *dstIt);
			dstIt->angle = srcIt->angle;
			dstIt->class_id = srcIt->class_id;
		}
    }
}

static void filterEllipticKeyPointsByImageSize( vector<EllipticKeyPoint>& keypoints, const Size& imgSize )
{
    if( !keypoints.empty() )
    {
        vector<EllipticKeyPoint> filtered;
        filtered.reserve(keypoints.size());
        vector<EllipticKeyPoint>::const_iterator it = keypoints.begin();
        for( int i = 0; it != keypoints.end(); ++it, i++ )
        {
            if( it->center.x + it->boundingBox.width < imgSize.width &&
                it->center.x - it->boundingBox.width > 0 &&
                it->center.y + it->boundingBox.height < imgSize.height &&
                it->center.y - it->boundingBox.height > 0 ){
					EllipticKeyPoint a = *it;
					a.angle = it->angle;
					a.class_id = it->class_id;
                filtered.push_back(a);
			}
        }
        keypoints.assign(filtered.begin(), filtered.end());
		for(int i = 0;i < keypoints.size();i++){
			keypoints[i].angle = filtered[i].angle;
			keypoints[i].class_id = filtered[i].class_id;
		}
    }
}

struct IntersectAreaCounter
{
    IntersectAreaCounter( float _dr, int _minx, int _maxx,
                          int _miny, int _maxy,
                          const Point2f& _diff,
                          const Scalar& _ellipse1, const Scalar& _ellipse2 ) :
               dr(_dr), bua(0), bna(0), minx(_minx),  maxx(_maxx), miny(_miny), maxy(_maxy),
               diff(_diff), ellipse1(_ellipse1), ellipse2(_ellipse2) {}

    void calc()
    {
        CV_Assert( miny < maxy );
        CV_Assert( dr > FLT_EPSILON );
        
        bua = 0;
        bna = 0;
        int temp_bua = bua, temp_bna = bna;

		for( float rx1 = (float)minx; rx1 <= (float)maxx; rx1 += dr )
		{ 
			float rx2 = rx1 - diff.x;
            for( float ry1 = (float)miny; ry1 <= (float)maxy; ry1 += dr )
            {
                float ry2 = ry1 - diff.y;
                //compute the distance from the ellipse center
                float e1 = (float)(ellipse1[0]*rx1*rx1 + 2*ellipse1[1]*rx1*ry1 + ellipse1[2]*ry1*ry1);
                float e2 = (float)(ellipse2[0]*rx2*rx2 + 2*ellipse2[1]*rx2*ry2 + ellipse2[2]*ry2*ry2);
                //compute the area
                if( e1<1 && e2<1 ) temp_bna++;
                if( e1<1 || e2<1 ) temp_bua++;
            }
        }
        bua = temp_bua;
        bna = temp_bna;
    }

    void join( IntersectAreaCounter& ac )
    {
        bua += ac.bua;
        bna += ac.bna;
    }

    float dr;
    int bua, bna;

    int minx, maxx;
    int miny, maxy;

    Point2f diff;
    Scalar ellipse1, ellipse2;

};

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



static void computeOneToOneMatchedOverlaps(const Mat & H , 
	const vector<EllipticKeyPoint>& keypoints1, const vector<EllipticKeyPoint>& keypoints2t,
											 vector<KeyPoint>& _keypoints1, vector<KeyPoint>& _keypoints2,
                                            bool commonPart, vector<SIdx>& overlaps, float minOverlap )
{
    CV_Assert( minOverlap >= 0.f );
    overlaps.clear();
    if( keypoints1.empty() || keypoints2t.empty() )
        return;

    overlaps.clear();

    for( size_t i1 = 0; i1 < keypoints1.size(); i1++ )
    {
        EllipticKeyPoint kp1 = keypoints1[i1];
        float maxDist = sqrt(kp1.axes.width*kp1.axes.height),
              fac = 30.f/maxDist;
        if( !commonPart )
            fac=3;

        maxDist = maxDist*4;
        fac = 1.f/(fac*fac);

        EllipticKeyPoint keypoint1a = EllipticKeyPoint( kp1.center, Scalar(fac*kp1.ellipse[0], fac*kp1.ellipse[1], fac*kp1.ellipse[2]) );

        for( size_t i2 = 0; i2 < keypoints2t.size(); i2++ )
        {
            EllipticKeyPoint kp2 = keypoints2t[i2];
            Point2f diff = kp2.center - kp1.center;

			if( norm(diff) < maxDist )
            {
                EllipticKeyPoint keypoint2a = EllipticKeyPoint( kp2.center, Scalar(fac*kp2.ellipse[0], fac*kp2.ellipse[1], fac*kp2.ellipse[2]) );
                //find the largest eigenvalue
                int maxx =  (int)ceil(( keypoint1a.boundingBox.width > (diff.x+keypoint2a.boundingBox.width)) ?
                                     keypoint1a.boundingBox.width : (diff.x+keypoint2a.boundingBox.width));
                int minx = (int)floor((-keypoint1a.boundingBox.width < (diff.x-keypoint2a.boundingBox.width)) ?
                                    -keypoint1a.boundingBox.width : (diff.x-keypoint2a.boundingBox.width));

                int maxy =  (int)ceil(( keypoint1a.boundingBox.height > (diff.y+keypoint2a.boundingBox.height)) ?
                                     keypoint1a.boundingBox.height : (diff.y+keypoint2a.boundingBox.height));
                int miny = (int)floor((-keypoint1a.boundingBox.height < (diff.y-keypoint2a.boundingBox.height)) ?
                                    -keypoint1a.boundingBox.height : (diff.y-keypoint2a.boundingBox.height));
                int mina = (maxx-minx) < (maxy-miny) ? (maxx-minx) : (maxy-miny) ;

                //compute the area
                float dr = (float)mina/50.f;
                int N = (int)floor((float)(maxx - minx) / dr);
                IntersectAreaCounter ac( dr, minx, maxx, miny, maxy, diff, keypoint1a.ellipse, keypoint2a.ellipse );
				ac.calc();
                if( ac.bna > 0 )
                {
                    float ov =  (float)ac.bna / (float)ac.bua;

					// ************************************************************************
					// orientation correction
					float t1 = M_PI * kp1.angle / 180.0;
					float t2 = M_PI * kp2.angle / 180.0;
					float tx1 = cos(t1), ty1 = sin(t1);
					float tx2 = cos(t2), ty2 = sin(t2);
					float tx3,ty3, tx0, ty0;
					float rtx, rty, rtz;
					float rtx0, rty0, rtz0;
					rtx0 = H.at<float>(0,2);
					rty0 = H.at<float>(1,2);
					rtz0 = H.at<float>(2,2);
					rtx = H.at<float>(0,0)*tx1+H.at<float>(0,1)*ty1 + H.at<float>(0,2);
					rty = H.at<float>(1,0)*tx1+H.at<float>(1,1)*ty1 + H.at<float>(1,2);
					rtz = H.at<float>(2,0)*tx1+H.at<float>(2,1)*ty1 + H.at<float>(2,2);
					tx3 = rtx / rtz;
					ty3 = rty / rtz;
					tx0 = rtx0 / rtz0;
					ty0 = rty0 / rtz0;
					tx3 = tx3 - tx0;
					ty3 = ty3 - ty0;
					float rd = sqrt(tx3*tx3+ty3*ty3);
					tx3 /= rd;
					ty3 /= rd;
					float dd = tx2*tx3 + ty2*ty3;
					// ************************************************************************
					// To find correct correspondings, we evaluate center distance and angle diffrence.
					ov = ov + dd * 0.000001;
					if(norm(diff) < 5.0){
						ov = ov + (5.0 - norm(diff))*0.01;
					}
					if( ov >= minOverlap || norm(diff) < 3.0){
						overlaps.push_back(SIdx(ov, (int)i1, (int)i2));
					}
				}
			}
		}
    }

    sort( overlaps.begin(), overlaps.end() );

    typedef vector<SIdx>::iterator It;

    It pos = overlaps.begin();
    It end = overlaps.end();

    while(pos != end)
    {
        It prev = pos++;
     
		end = std::remove_if(pos, end, SIdx::UsedFinder(*prev));
    }
    overlaps.erase(pos, overlaps.end());
}

static void calculateRepeatability( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                    vector<KeyPoint>& _keypoints1, vector<KeyPoint>& _keypoints2,
                                    float& repeatability, int& correspondencesCount,  vector<SIdx> & overlaps,
                                    Mat* thresholdedOverlapMask=0  )
{
    vector<EllipticKeyPoint> keypoints1, keypoints2, keypoints1t, keypoints2t;
    EllipticKeyPoint::convert( _keypoints1, keypoints1 );
    EllipticKeyPoint::convert( _keypoints2, keypoints2 );

    // calculate projections of key points
    EllipticKeyPoint::calcProjection( keypoints1, H1to2, keypoints1t );
    Mat H2to1; invert(H1to2, H2to1);
    EllipticKeyPoint::calcProjection( keypoints2, H2to1, keypoints2t );

    float overlapThreshold;
    bool ifEvaluateDetectors = thresholdedOverlapMask == 0;
    if( ifEvaluateDetectors )
    {
        overlapThreshold = 1.f - 0.4f;

        // remove key points from outside of the common image part
        Size sz1 = img1.size(), sz2 = img2.size();
		
        filterEllipticKeyPointsByImageSize( keypoints1, sz1 );
        filterEllipticKeyPointsByImageSize( keypoints1t, sz2 );
        filterEllipticKeyPointsByImageSize( keypoints2, sz2 );
        filterEllipticKeyPointsByImageSize( keypoints2t, sz1 );

    }
    else
    {
        overlapThreshold = 1.f - 0.5f;

        thresholdedOverlapMask->create( (int)keypoints1.size(), (int)keypoints2t.size(), CV_8UC1 );
        thresholdedOverlapMask->setTo( Scalar::all(0) );
    }
    size_t size1 = keypoints1.size(), size2 = keypoints2t.size();
    size_t minCount = MIN( size1, size2 );

    // calculate overlap errors
    computeOneToOneMatchedOverlaps(H1to2,  keypoints1, keypoints2t, _keypoints1, _keypoints2, ifEvaluateDetectors, overlaps, overlapThreshold/*min overlap*/ );

    correspondencesCount = -1;
    repeatability = -1.f;
    if( overlaps.empty() )
        return;

    if( ifEvaluateDetectors )
    {
        // regions one-to-one matching
        correspondencesCount = (int)overlaps.size();
        repeatability = minCount ? (float)correspondencesCount / minCount : -1;
    }
    else
    {
        for( size_t i = 0; i < overlaps.size(); i++ )
        {
            int y = overlaps[i].i1;
            int x = overlaps[i].i2;
            thresholdedOverlapMask->at<uchar>(y,x) = 1;
        }
    }  
	EllipticKeyPoint::calcProjection( keypoints2t, H1to2, keypoints2 );
	EllipticKeyPoint::convert( keypoints2,  _keypoints2 );
	EllipticKeyPoint::convert( keypoints1,  _keypoints1 );
}

void evaluateFeatureDetector2( const Mat& img1, const Mat& img2, const Mat& H1to2,
                              vector<KeyPoint>* _keypoints1, vector<KeyPoint>* _keypoints2,
                              float& repeatability, int& correspCount, vector<SIdx> & overlaps)
{
    calculateRepeatability( img1, img2, H1to2, *_keypoints1, *_keypoints2, repeatability, correspCount,overlaps );
}