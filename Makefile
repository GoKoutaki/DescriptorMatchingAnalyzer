CC	= gcc
CPPFLAGS	= -O3 -I/usr/local/include/opencv/ -L/usr/local/lib/opencv/
LDFLAGS	=
LIBS	= -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_calib3d  -lopencv_features2d -lopencv_nonfree
TARGET	= main.exe
OBJS	= main.o evaluation.o
all:	$(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm *.o
	