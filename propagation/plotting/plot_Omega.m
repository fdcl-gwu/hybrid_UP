function [  ] = plot_Omega( path, L, sf, slow )

% count number of piece of data
files = dir(path);
fnt = zeros(length(files),1);
fxnt = zeros(length(files),1);
Nf = 0;
Nfx = 0;
for i = 1:length(files)
    if contains(files(i).name,'fx')
        ind_dot = strfind(files(i).name,'.');
        nt = str2double(files(i).name(3:ind_dot-1));
        Nfx = Nfx+1;
        fxnt(Nfx) = nt;
    elseif contains(files(i).name,'f')
        ind_dot = strfind(files(i).name,'.');
        nt = str2double(files(i).name(2:ind_dot-1));
        Nf = Nf+1;
        fnt(Nf) = nt;
    end
end

if Nfx > 0
    have_fx = true;
    N = Nfx;
    fnt = fxnt;
else
    have_fx = false;
    N = Nf;
end

fnt(N+1:end) = [];
fnt = sort(fnt);

% grid
if have_fx
    load(strcat(path,'/fx1'),'fx');
    Bx = size(fx,2)/2;
    x = linspace(-L/2,L/2-L/(2*Bx),2*Bx);
    y = x;
else
    load(strcat(path,'/f1'),'f');
    BR = size(f,1)/2;
    Bx = size(f,4)/2;
    x = linspace(-L/2,L/2-L/(2*Bx),2*Bx);
    y = x;

    beta = pi/(4*BR)*(2*(0:(2*BR-1))+1);
    w = zeros(1,2*BR);
    for j = 1:2*BR
        w(j) = 1/(4*BR^3)*sin(beta(j))*sum(1./(2*(0:BR-1)+1).*sin((2*(0:BR-1)+1)*beta(j)));
    end
end

% find the largest fx
try
    load(strcat(path,'/fx_max.mat'),'fx_max');
catch
    fx_max = zeros(N,1);
    N = 0;
    for nt = fnt'
        if have_fx
            if nt > 1
                load(strcat(path,'/fx',num2str(nt)),'fx');
            end
        else
            if nt > 1
                load(strcat(path,'/f',num2str(nt)),'f');
            end
            fx = permute(sum(f.*w,[1,2,3]),[4,5,1,2,3]);
            save(strcat(path,'/fx',num2str(nt)),'fx');
        end

        N = N+1;
        fx_max(N) = max(fx,[],'all');
    end
    fx_max = max(fx_max);
    save(strcat(path,'/fx_max.mat'),'fx_max');
end

% plot
N = 0;
for nt = fnt'
    load(strcat(path,'/fx',num2str(nt)),'fx');
    if size(fx,1) == 1
        fx = permute(fx,[2,3,1]);
    end
    fx(fx<0) = 0;
    
    fig = figure; hold on;
    surf(x,y,fx,'LineStyle','none');
    caxis([0,fx_max]);
    xlim([-1.7,1.7]);
    ylim([-1.7,1.7]);
    xticks([-1.7,0,1.7]);
    yticks([-1.7,0,1.7]);
    xticklabels({'-14.5','0','14.5'});
    yticklabels({'-14.5','0','14.5'});
    zlim([0,4.5]);
    xlabel('$\Omega_2$ (deg/s)','Interpreter','latex','Position',[0.250,1.784,-0.442]);
    ylabel('$\Omega_1$ (deg/s)','Interpreter','latex','Position',[1.831,0.203,-0.443]);
    view([1,1,1]);
    
    annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),...
        'Position',[0.13,0.78,0.22,0.07],'LineStyle','none');
    
    title('Marginal angular velocity density');
    
    N = N+1;
    M(N) = getframe(fig);
    close(fig);
end

% generate video
v = VideoWriter(strcat(path,'/Omega.avi'));

if exist('slow','var')
    v.FrameRate = sf/(fnt(2)-fnt(1))/slow;
else
    v.FrameRate = sf;
end
v.Quality = 100;
open(v);
writeVideo(v,M);
close(v);

end

