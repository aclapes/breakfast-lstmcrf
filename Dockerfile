FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu14.04
MAINTAINER aclapes

RUN echo "deb http://us.archive.ubuntu.com/ubuntu trusty main multiverse" >> /etc/apt/sources.list \
	&& apt-get update -qq \
 	&& apt-get install --no-install-recommends -y \
   	# install essentials
	build-essential \ 
	git \
	wget \ 
	nano \
	# install python 2 
	python \ 
	python-dev \ 
	python-pip \ 
	python-wheel \ 
	pkg-config \ 
	libpng-dev \
	zlib1g-dev \
	unzip \
	ssh \
	openssh-server \ 
	cmake yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine-dev gstreamer1.0 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils \
	&& apt-get clean \ 
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools pip
RUN pip install --upgrade \
    imageio \
    h5py \
    jupyter==1.0.0 \
    numpy \
    scipy \
    progressbar2==3.9.3 \
    youtube-dl \
    tensorflow-gpu==1.4.0

RUN mkdir opt/externals

RUN cd opt/externals \
	&& git clone -b n3.0 git://source.ffmpeg.org/ffmpeg.git ffmpeg \
	&& cd ffmpeg \
	&& ./configure --enable-nonfree --enable-pic --enable-shared \
	&& make -j40 \
	&& make install \
	&& echo "/usr/local/lib/" >> /etc/ld.so.conf \
	&& ldconfig

RUN cd opt/externals \
	&& git clone -b 2.4.13.2 https://github.com/opencv/opencv.git opencv \
	&& cd opencv \
	&& mkdir build \
	&& cd build \
	&& cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
		-D WITH_TBB=ON \
		-D BUILD_NEW_PYTHON_SUPPORT=ON \
		-D WITH_V4L=ON \
		-D WITH_CUDA=ON \
		-D WITH_QT=ON \ 
		-D WITH_OPENGL=ON .. \
	&& make -j40 \
	&& make install


# Create user for ssh connection
RUN useradd -ms /bin/bash dockeruser && \
        echo dockeruser:dockerpass | chpasswd && \
        usermod -a -G sudo dockeruser && \
        echo "AllowUsers dockeruser" >> /etc/ssh/sshd_config

# Make user's source and copy code into it
# RUN mkdir /home/dockeruser/src && \
#        cd /home/dockeruser/src && \
#	git clone https://github.com/aclapes/breakfast-lstmcrf.git && \
#	chown dockeruser:dockeruser -R /home/dockeruser/src
COPY . /home/dockeruser/src
RUN mkdir /home/dockeruser/log && chown dockeruser:dockeruser -R /home/dockeruser/

# Open port 22 to map with some host's port
EXPOSE 22
RUN mkdir /var/run/sshd
EXPOSE 6006

RUN sh -c 'ln -s /dev/null /dev/raw1394'

CMD /usr/sbin/sshd && bash
