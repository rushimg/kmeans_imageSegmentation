I1= imread('testImageGrey.bmp');

I=double(I1);

figure
subplot(2,2,1);
imshow(I1);

subplot(2,2,2)
%imshow(I2);
[g,C]=kmeans(I(:),8);
J = reshape(g,size(I));
subplot(2,2,3);
disp(C);
imshow(J,[]);
