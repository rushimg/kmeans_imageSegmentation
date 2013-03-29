I1= imread('sg.jpg');
I2=rgb2gray(I1);
%I=double(I2);

%figure
%subplot(2,2,1);
%imshow(I1);

%subplot(2,2,2)
%imshow(I2);
%g=kmeans(I(:),4);
%J = reshape(g,size(I));
%subplot(2,2,3);
%imshow(J,[]);

dlmwrite('sg_gs.txt', I2);
dlmwrite('sg_rgb.txt', I1);