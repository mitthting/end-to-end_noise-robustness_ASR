% x = linspace(0,10,150);
% y = cos(5*x);
% 
% figure
% plot(x,y,'Color',[0,0.7,0.9])
% 
% title('2-D Line Plot')
% xlabel('x')
% ylabel('cos(5x)')

% %--------------------------------clean-------------------------------
% filec = fopen('/home/ding/data/DAE/clean.txt','r');
% 
% formatSpec = '%f';
% 
% snrc = fscanf(filec,formatSpec);
% 
% fclose(filec);

%--------------------------------SNR20-------------------------------
filec = fopen('/home/ding/data/DAE/SNR20.txt','r');

formatSpec = '%f';

snr20 = fscanf(filec,formatSpec);

fclose(filec);


%--------------------------------SNR15-------------------------------
filec = fopen('/home/ding/data/DAE/SNR15.txt','r');

formatSpec = '%f';

snr15 = fscanf(filec,formatSpec);

fclose(filec);



%--------------------------------SNR10-------------------------------
filec = fopen('/home/ding/data/DAE/SNR10.txt','r');

formatSpec = '%f';

snr10 = fscanf(filec,formatSpec);

fclose(filec);


%--------------------------------SNR5-------------------------------
filec = fopen('/home/ding/data/DAE/SNR5.txt','r');

formatSpec = '%f';

snr5 = fscanf(filec,formatSpec);

fclose(filec);

%Y = [snrc snr20 snr15 snr10 snr5];
Y = [snr20 snr15 snr10 snr5];

%Y([101],:) = [];


figure
plot( Y )
 
xlabel('epoch')
ylabel('weight')
 
 

% alphac = 0.1;
% alpha5 = 0.1;
% alpha10 = 0.1;
% alpha15 = 0.1;
% alpha20 = 0.1;
% 
% a=[];
% b=[];
% c=[];
% d=[];
% e=[];
% for i =1:100
%     a(i) = alpha5 ;
%     
%     alpha5 = alpha5 * 0.99;
% 
%     b(i) = alpha10 ;
%     
%     alpha10 = alpha10 * 0.98;
%     
%     c(i) = alpha15 ;
%     
%     alpha15 = alpha15 * 0.97;
%     
%     d(i) = alpha20 ;
%     
%     alpha20 = alpha20 * 0.96;
%     
%     e(i) = alphac ;
%     
%     alphac = alphac * 0.95;
%     
% end
% 
% Y = [e' d' c' b' a'];
% 
%  figure
%  plot( Y )
%   
%  xlabel('epoch')
%  ylabel('value')

