%
%

close all
clear

global fullImageName
path = 'NemaLife Images_Converted/'
fullImageName = '00106';

I1=imread(strcat(path,fullImageName,'.jpg'));
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
% figure; imshow(I1,[])
% figure; imshow(th,[])
% figure; imshow(imoverlay(I1,th,'r'),[])

[outL,outN]=bwlabel(th);

fstats=regionprops('table',outL,'Area','BoundingBox');
idx = find([fstats.Area] >200);
fstats = fstats(idx,:)
bboxes=fstats.BoundingBox;

Things = insertShape(I1,'Rectangle',bboxes,'LineWidth',3);
% figure; imshow(Things,[]);
%%
% xy=fstats{1}; % Get n by 2 array of x,y coordinates.
% x = xy(:, 2); % Columns. 
% y = xy(:, 1); % Rows.
% leftColumn = min(x);
% rightColumn = max(x);
% topLine = min(y);
% bottomLine = max(y);
% width = rightColumn - leftColumn + 1;
% height = bottomLine - topLine + 1;

global i
w = wormGUI()
for i = 1:height(fstats)
    len = max(fstats(i,:).BoundingBox(3),fstats(i,:).BoundingBox(4));
    s = fstats(i,:).BoundingBox;
    s(3) = len;
    s(4) = len;
    croppedImage = imcrop(I1, s);
    updateimage(w,croppedImage);
    global busy
    while busy == true
        pause(.1)
    end
end