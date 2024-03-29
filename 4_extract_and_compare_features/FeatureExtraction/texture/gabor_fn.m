function [gbr gbi] = gabor_fn(sigma,theta,lambda,psi,gamma,nstds)
 
% sigma = 0.5;
% theta = 0;
% lambda = 0.5;
% psi = 0;
% gamma = 1;
% nstds = 3;

sigma_x = sigma;
sigma_y = sigma/gamma;
 
% Bounding box
xmax = max(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
xmax = ceil(max(1,xmax));
ymax = max(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
ymax = ceil(max(1,ymax));
xmin = -xmax; ymin = -ymax;
[x,y] = meshgrid(xmin:xmax,ymin:ymax);
 
% Rotation 
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);
 

% Real
gbr = exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);
% Imaginary
gbi = exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*sin(2*pi/lambda*x_theta+psi);

