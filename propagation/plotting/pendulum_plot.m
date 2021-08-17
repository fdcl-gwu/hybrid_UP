function [  ] = pendulum_plot( varargin )

close all;

addpath('../rotation3d');

if (size(varargin{1},1)==3 && size(varargin{1},2)==3)
    isDensity = false;
    R = varargin{1};
    sf = varargin{2};
    
    if size(varargin,2) >= 3
        slow = varargin{3};
    end
else
    isDensity = true;
    path = varargin{1};
    L = varargin{2};
    sf = varargin{3};
    
    if size(varargin,2) >= 4
        slow = varargin{4};
    end
    
    if size(varargin,2) >= 5
        use_mex = varargin{5};
    else
        use_mex = false;
    end
    
    if use_mex
        addpath('mex');
    end
end

if isDensity
    files = dir(path);
    
    Nt = 0;
    for i = 1:length(files)
        if strcmp(files(i).name(1),'f')
            ind_dot = strfind(files(i).name,'.');
            nt = str2double(files(i).name(2:ind_dot-1));
            if nt > Nt
                Nt = nt;
            end
        end
    end
    
    if Nt > 0
        f = load(strcat(path,'/f',num2str(1)));
        f = f.f;
        
        % SO(3) grid
        BR = size(f,1)/2;
        Bx = size(f,4)/2;
        R = RGrid(BR);
        
        % spherical grid
        Nt1 = 100;
        Nt2 = 50;
        theta1 = linspace(-pi,pi,Nt1);
        theta2 = linspace(0,pi,Nt2);
        s1 = cos(theta1)'.*sin(theta2);
        s2 = sin(theta1)'.*sin(theta2);
        s3 = repmat(cos(theta2),Nt1,1);
        
        % circular grid
        Nalpha = 40;
        alpha = linspace(-pi,pi-2*pi/Nalpha,Nalpha);
        
        if ~use_mex
            % 3d interpolation
            d_threshold = 0.4;

            try
                ind_R = load(strcat(path,'/ind_R'));
                ind_R = ind_R.ind_R;
                v_R = load(strcat(path,'/v_R'));
                v_R = v_R.v_R;
            catch
                [ind_R,v_R] = interpInd(R,s1,s2,s3,alpha,d_threshold);
                save(strcat(path,'/ind_R'),'ind_R','-v7.3');
                save(strcat(path,'/v_R'),'v_R','-v7.3');
            end
        else
            e = get_Euler(reshape(s1,[],1),reshape(s2,[],1),reshape(s3,[],1),alpha);
            
            v_R = [];
            ind_R = [];
        end
        
        if ~use_mex
            parfor nt = 1:Nt
                f = load(strcat(path,'/f',num2str(nt)));
                f = f.f;

                if ~isa(f,'double')
                    f = double(f);
                end

                if ndims(f) == 6
                    fR = sum(f,[4,5,6])*(L/(2*Bx))^3;
                elseif ndims(f) == 5
                    fR = sum(f,[4,5])*(L/(2*Bx))^2;
                else
                    error('Dimensions of f is wrong');
                end
                fR = reshape(fR,1,[]);

                c = zeros(Nt1,Nt2,3);
                for nt1 = 1:Nt1
                    for nt2 = 1:Nt2
                        for i = 1:3
                            for ntheta = 1:Nalpha
                                v = v_R{nt1,nt2,ntheta,i};
                                ind = ind_R{nt1,nt2,ntheta,i};
                                if size(v,1) > 1
                                    f_interp = scatteredInterpolant(v',fR(ind_R{nt1,nt2,ntheta,i})');
                                    c(nt1,nt2,i) = c(nt1,nt2,i) + f_interp(zeros(1,size(v,1)));
                                else
                                    [v,I] = sort(v);
                                    ind = ind(I);
                                    c(nt1,nt2,i) = c(nt1,nt2,i) + interp1(v,fR(ind),0);
                                end
                            end
                        end
                    end
                end

                c = c/max(c(:));
                c1 = c(:,:,1);
                c2 = c(:,:,2);
                c3 = c(:,:,3);
                c = ones(size(c));
                c(:,:,1) = c(:,:,1)-c2-c3;
                c(:,:,2) = c(:,:,2)-c1-c3;
                c(:,:,3) = c(:,:,3)-c1-c2;

                f = figure;
                surf(s1,s2,s3,c,'LineStyle','none','FaceColor','interp');

                xlim([-1,1]);
                ylim([-1,1]);
                zlim([-1,1]);
                view([1,-1,0]);
                axis equal;

                annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),'Position',[0.15,0.75,0.16,0.07]);

                M(nt) = getframe;
                close(f);
            end
        else
            for nt = 1:Nt
                tic;
                f = load(strcat(path,'/f',num2str(nt)));
                f = f.f;

                if ~isa(f,'double')
                    f = double(f);
                end
                
                c = getColor_mex(f,e,L);
                c = reshape(c,Nt1,Nt2,3);
                
                c = c/max(c(:));
                c1 = c(:,:,1);
                c2 = c(:,:,2);
                c3 = c(:,:,3);
                c = ones(size(c));
                c(:,:,1) = c(:,:,1)-c2-c3;
                c(:,:,2) = c(:,:,2)-c1-c3;
                c(:,:,3) = c(:,:,3)-c1-c2;

                f = figure;
                surf(s1,s2,s3,c,'LineStyle','none','FaceColor','interp');

                xlim([-1,1]);
                ylim([-1,1]);
                zlim([-1,1]);
                view([0,-1,-1]);
                axis equal;

                annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),'Position',[0.15,0.75,0.16,0.07]);

                M(nt) = getframe;
                close(f);
                save(strcat(path,'/c',num2str(nt)),'c');
                toc;
                disp(strcat('f',num2str(nt),' finished'));
            end
        end
    end
else
    if ndims(R) == 3
        Nt = size(R,3);
    else
        Nt = size(R,4);
    end
    
    for nt = 1:Nt
        f = figure; hold on;
        color = get(gca,'ColorOrder');
        
        if ndims(R) == 3
            plot3([0,R(1,1,nt)],[0,R(2,1,nt)],[0,R(3,1,nt)]);
            plot3([0,R(1,2,nt)],[0,R(2,2,nt)],[0,R(3,2,nt)]);
            plot3([0,R(1,3,nt)],[0,R(2,3,nt)],[0,R(3,3,nt)]);
        else
            for i = 1:size(R,3)
                plot3([0,R(1,1,i,nt)],[0,R(2,1,i,nt)],[0,R(3,1,i,nt)],'Color',color(1,:));
                plot3([0,R(1,2,i,nt)],[0,R(2,2,i,nt)],[0,R(3,2,i,nt)],'Color',color(2,:));
                plot3([0,R(1,3,i,nt)],[0,R(2,3,i,nt)],[0,R(3,3,i,nt)],'Color',color(3,:));
            end
        end

        xlim([-1,1]);
        ylim([-1,1]);
        zlim([-1,1]);
        view([0,0,1]);

        annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),'Position',[0.15,0.85,0.16,0.07]);

        M(nt) = getframe;
        close(f);
    end
end

if isDensity
    v = VideoWriter(strcat(path,'/R1.avi'));
else
    v = VideoWriter('R1.avi');
end

if exist('slow','var')
    v.FrameRate = sf/slow;
else
    v.FrameRate = sf;
end
v.Quality = 100;
open(v);
writeVideo(v,M);
close(v);

rmpath('../rotation3d');
if isDensity && use_mex
    rmpath('mex');
end

end


function [ R ] = RGrid( BR )

alpha = reshape(pi/BR*(0:(2*BR-1)),1,1,[]);
beta = reshape(pi/(4*BR)*(2*(0:(2*BR-1))+1),1,1,[]);
gamma = reshape(pi/BR*(0:(2*BR-1)),1,1,[]);

ca = cos(alpha);
sa = sin(alpha);
cb = cos(beta);
sb = sin(beta);
cg = cos(gamma);
sg = sin(gamma);

Ra = [ca,-sa,zeros(1,1,2*BR);sa,ca,zeros(1,1,2*BR);zeros(1,1,2*BR),zeros(1,1,2*BR),ones(1,1,2*BR)];
Rb = [cb,zeros(1,1,2*BR),sb;zeros(1,1,2*BR),ones(1,1,2*BR),zeros(1,1,2*BR);-sb,zeros(1,1,2*BR),cb];
Rg = [cg,-sg,zeros(1,1,2*BR);sg,cg,zeros(1,1,2*BR);zeros(1,1,2*BR),zeros(1,1,2*BR),ones(1,1,2*BR)];

R = zeros(3,3,2*BR,2*BR,2*BR);
for i = 1:2*BR
    for j = 1:2*BR
        for k = 1:2*BR
            R(:,:,i,j,k) = Ra(:,:,i)*Rb(:,:,j)*Rg(:,:,k);
        end
    end
end

R = reshape(R,3,3,(2*BR)^3);

end


function [ ind_R, v_R ] = interpInd( R, s1, s2, s3, theta, d )

Nt1 = size(s1,1);
Nt2 = size(s1,2);
Ntheta = size(theta,2);

ind_R = cell(Nt1,Nt2,Ntheta,3);
v_R = cell(Nt1,Nt2,Ntheta,3);

for i = 1:3
    vref = [0;0;0];
    vref(i) = 1;

    for nt1 = 1:Nt1
        for nt2 = 1:Nt2
            rref = [s1(nt1,nt2);s2(nt1,nt2);s3(nt1,nt2)];

            Rref = eye(3);
            jk = setdiff(1:3,i);
            Rref(:,i) = rref;
            Rref(:,jk) = null(rref');
            D = eye(3);
            D(jk(1),jk(1)) = det(Rref);
            Rref = Rref*D;
            Rref = mulRot(Rref,expRot(vref.*theta));

            for ntheta = 1:Ntheta
                v = logRot(mulRot(Rref(:,:,ntheta)',R),'v');
                ind_R{nt1,nt2,ntheta,i} = find(sqrt(sum(v.^2)) < d);
                v = v(:,ind_R{nt1,nt2,ntheta,i});
                            
                v_isunique = false(3,1);
                for j = 1:3
                    v_isunique(j) = max(v(j,:))-min(v(j,:)) < 1e-14;
                end

                v(v_isunique,:) = [];
                v_R{nt1,nt2,ntheta,i} = v;
            end
        end
    end
end

end


function [ e ] = get_Euler( s1, s2, s3, alpha )

Ns = length(s1);
Na = length(alpha);

R = zeros(3,3,Ns,Na,3);
for ns = 1:Ns
    rref = [s1(ns);s2(ns);s3(ns)];

    for nd = 1:3
        Rref = eye(3);
        jk = setdiff(1:3,nd);
        Rref(:,nd) = rref;
        Rref(:,jk) = null(rref');
        D = eye(3);
        D(jk(1),jk(1)) = det(Rref);
        Rref = Rref*D;

        vref = zeros(3,Na);
        vref(nd,:) = alpha;
        R(:,:,ns,:,nd) = mulRot(Rref,expRot(vref));
    end
end

e = rot2eul(reshape(R,[3,3,Ns*Na*3]),'zyz');
e = reshape(e,[3,Ns,Na,3]);

end

