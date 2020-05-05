%
%

close all
clear

I1=imread('NemaLife Images_Converted/00104.jpg');
I1=rgb2gray(I1);
figure; imshow(I1,[])

[M,N]=size(I1);
[U,V]=meshgrid([1:N],[1:M]);
D= sqrt((U-(N+1)/2).^2+(V-(M+1)/2).^2);
D0=2;
n=2; 
one=ones(M,N);
H = 1./(one+(D./D0).^(2*n));
G=fftshift(fft2(I1)).*H;
g=real(ifft2(ifftshift(G)));
out=double(I1)-g;
I1=uint8((255.0/(max(out(:))-min(out(:)))).*(out-min(out(:))));

th=imbinarize(I1,'Adaptive','Sensitivity',0.4);
figure; imshow(I1,[])
figure; imshow(th,[])
figure; imshow(imoverlay(I1,th,'r'),[])

[outL,outN]=bwlabel(th);

fstats=regionprops('table',outL,'Area','BoundingBox');
bboxes=fstats.BoundingBox;

Things = insertShape(I1,'Rectangle',bboxes,'LineWidth',3);
figure; imshow(Things,[]);
