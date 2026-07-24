%--------------------------------
% function floodplain_wrapper_<AOI>(typ, nc0 <optional input>)
%--------------------------------
% This function creates a stream network and applies the FLDPLN model to
%  create a SLIE.  Modify parameters in the code to customize.
%
% INPUT
%  typ = 'hd', 'ram0', or 'ram'
%        NOTE: 'hd' uses minimal RAM, 'ram0' uses medium RAM, and 'ram'
%              uses the most RAM (nearly twice as much as 'ram0')
% lib0 = 'verdigris', 'neosho', 'osage', 'ksmo', 'bigblue', 'republican',
%        'smoky', or 'arkcim'
%
% OPTIONAL INPUT
%  nc0 = user-specified size of worker parallel pool, which must be an
%        integer between 1 and the number of physical cores available
%--------------------------------
%--------------------------------
function fldpln_wrapper_v8_ks2024(typ,lib0,varargin)
%--------------------------------

if(~ismember(typ,{'hd','ram0','ram'}))
   error('''typ'' must be ''hd'', ''ram0'', or ''ram''');
else
   typnum = find(ismember({'hd','ram0','ram'},typ));
end

libs = {'verdigris','neosho','osage','ksmo','bigblue','republican','smoky','arkcim'};
if(~ismember(lib0,libs))
   error('unknown library')
end

% check for optional input (user-specified size of worker parallel pool)
if(nargin==3)
   nc = feature('numcores');
   nc0 = varargin{1};
   if(~ismember(nc0,1:nc))
      error(['specified number of cores exceed the maximum of ',int2str(nc)])
   end
end
%=============================
% Enter home directory
%-----------------------------
dr1 = lib0;
dr0 = ['d:/floodplain/ks2024/libraries/',dr1,'/'];
if(~exist(dr0,'dir'))
   dr0 = ['e:/floodplain/ks2024/libraries/',dr1,'/'];
   if(~exist(dr0,'dir'))
      error(['directory ''',dr0,''' does not exist'])
   end
end

%-----------------------------
% Initialize flood parameters & the I/O file names
%-----------------------------
fldmn = .05;
dh = 3; % step size
fldmx = 36; % max depth-to-flood (DTF)
mxht = 0;

demf = [dr0,'bil/dem.bil'];
filf = [dr0,'bil/fil.bil'];
fdrf = [dr0,'bil/fdr.bil'];
facf = [dr0,'bil/fac.bil'];
segf = [dr0,'bil/str_segid.bil'];
matf = [dr0,'mat/seg_info.mat'];
%shpf = [dr0,'vector/onion_start_pt.shp'];
shpf_all = [dr0,'vector/',lib0,'_str_segid_nogen_dissolve_mod4lib.shp'];

fdrinfo = readbilheader(fdrf);
pxsz = fdrinfo.pxszx;

%=============================
% Identify synthetic stream network
%-----------------------------
mi2px = 5280*.3048/pxsz; % horizontal units assumed to be meters (otherwise modify as needed) 
seglen = round(5*mi2px); % 5 miles maximum segment length
sqmi2px = (5280^2)*(.3048^2)/(pxsz^2); % horizontal units assumed to be meters (otherwise modify as needed)
facthr = round(25*sqmi2px); % 25 sq. mi minimum catchment size (min for segment break)
strthr = round(70*sqmi2px); % 70 sq. mi minimum catchment size (min for defining a stream)

if(~exist([dr0,'mat'],'dir'))
   mkdir([dr0,'mat']);
end

if(~exist(matf,'file'))
   create_segdb_facthr_maxlen(strthr,facthr,seglen,fdrf,facf,segf,matf)
%   create_segdb_facthr_maxlen_seedpts(shpf,facthr,seglen,fdrf,facf,segf,matf)
%   create_segdb_facthr_maxlen(strthr,strthr,seglen,fdrf,facf,segf,matf)
end

% replace FIL with masked version, if it exist, to avoid processing in
%  areas such as reservoirs
if(exist([dr0,'bil/fil_masked.bil'],'file'))
   filf = [dr0,'bil/fil_masked.bil']; % masked FIL to pass into FLDPLN
end

% Columns of 'seg_info' (row corresponds with segment ID):
% 1) start pixel (BIL vector file coordinate)
% 2) end pixel (BIL vector file coordinate)
% 3) FAC value @ start
% 4) FAC value @ end
% 5) segment length
% 6) immediately downstream segment ID (row index; 0 if none)
% 7) reach ID (from natural segmentation partition, i.e. hdwtr-confl & confl-confl)

%-----------------------------
% Define background value
[mn,mx,bg] = scanbil(filf);
%-----------------------------

load(matf,'seg_info');
num = size(seg_info,1);
seg_list = (1:num)';
try
   seg_list = sortrows(readshapefield(shpf_all,{'GRID_CODE'}));
catch
   seg_list = sortrows(readshapefield(shpf_all,{'grid_code'}));
end
num = length(seg_list);

%------------------
% Identify completed segments
if(1>0)
   dr = dir([dr0,'segment_files']);
   len = size(dr,1);
   done_segs = zeros(len-2,1);
   ct = 0;
   for j = 3:len
      f2 = find(dr(j).name=='.');
      if(dr(j).name(f2-1)=='p')
         load([dr0,'segment_files/',dr(j).name],'fldht')
         if(fldht==fldmx)
            f1 = find(dr(j).name=='g',1,'last');
            f2 = find(dr(j).name=='_',1,'last');
            done_segs(ct+1) = str2double(dr(j).name(f1+1:f2-1));
            ct = ct+1;
         end
      end
   end
   done_segs = done_segs(1:ct);
   seg_list = intersect(seg_list,setxor(seg_list,done_segs));
   num = length(seg_list);
%   return;
end
%------------------

%------------------
% run FLDPLN
%------------------
if(num>0)
   inp = struct;
   inp.bildr = [dr0,'bil/'];
   inp.segdr = [dr0,'segment_files/'];
   inp.dem = demf;
   inp.fil = filf;
   inp.fdr = fdrf;
   inp.seg = matf;
   fildat = [];
   fdrdat = [];

   % if 'ram' or 'ram0', load fildat & fdrdat
   if(typnum>1)
      %-------------------------
      % Make sure FIL & FDR have same row-col dimensions
      f = inp.fil;
      filinfo = readbilheader(f);
      r = filinfo.r;
      c = filinfo.c;
   
      f = inp.fdr;
      filinfo = readbilheader(f);
      r0 = filinfo.r;
      c0 = filinfo.c;
      
      if(or(r0~=r,c0~=c))
         error('FIL & FDR do not have the same row-col dimensions');
         % Write this message to a file in the home directory
      end
      %-------------------------
      
      %=============================
      % Read FIL file specs & data
      %-------------------------
      f = inp.fil;
      filinfo = readbilheader(f);
      fmt = filinfo.fmt;
      filtyp = filinfo.bytord;
   
      if(filtyp)
         fid = fopen(f,'rb','b');
      else
         fid = fopen(f,'rb');
      end
      fildat = zeros(r*c,1,fmt);
      fildat(1:r*c) = fread(fid,inf,fmt);
      fclose(fid);
      %=============================
      
      %=============================
      % Read FDR data
      %-------------------------
      f = inp.fdr;
      filinfo = readbilheader(f);
      fmt = filinfo.fmt;
      filtyp = filinfo.bytord;
   
      if(filtyp)
         fid = fopen(f,'rb','b');
      else
         fid = fopen(f,'rb');
      end
      fdrdat = zeros(r*c,1,fmt);
      fdrdat(1:r*c) = fread(fid,inf,fmt);
      fclose(fid);
      %=============================
   end

   if(1>0) % change to 'if(1<0)' if want to comment out & use 'parfor' loop instead
      
      delete(gcp('nocreate')); % deletes any existing worker parallel pool

      % check if optional user input of worker parallel pool size was specifed
      if(exist('nc0','var'))
         p = parpool(nc0);
      else % else define pool size to be min(max number of cores, number of segments)
         nc = feature('numcores');
         if(nc/3<=num)
            p = parpool;
         else
            p = parpool(num);
         end
      end
      
      for j = 1:num
         pf(j) = parfeval(p,@fldpln_model_v8,0,typ,seg_list(j),inp,fildat,fdrdat,fldmn,fldmx,dh,mxht,1,bg);
      end
      for idx = 1:num
     %    fetchNext blocks until next results are available.
         completedIdx = fetchNext(pf);
         fprintf('Segment %d completed\n', completedIdx);
      end
   else
      parfor j = 1:num
         fldpln_model_v8(typ,seg_list(j),inp,fildat,fdrdat,fldmn,fldmx,dh,mxht,1,bg);
      end
   end
end
%------------------
return;

% code to execute if want to raise fldmx without starting over
if(1>0)
   fs = int2str(fldmx);
   mkdir([dr0,'segment_files_h',fs]);
   copyfile([dr0,'segment_files/*.*'],[dr0,'segment_files_h',fs]);
   delete([dr0,'segment_files/*.*']);
   copyfile([dr0,'segment_files_h',fs,'/*tmp*'],[dr0,'segment_files']);
   fldmx = fldmx+15;
   batch_rename_segment_files([dr0,'segment_files/'],fldmx);
   return;
end

% Identify completed segments & write final file
if(1<0)
   dr = dir([dr0,'segment_files']);
   len = size(dr,1);
   done_segs = zeros(len-2,1);
   ct = 0;
   for j = 3:len
      f2 = find(dr(j).name=='.');
      if(dr(j).name(f2-1)=='p')
         load([dr0,'segment_files/',dr(j).name],'fldht')
         if(fldht==fldmx)
            f1 = find(dr(j).name=='g',1,'last');
            f2 = find(dr(j).name=='_',1,'last');
            done_segs(ct+1) = str2double(dr(j).name(f1+1:f2-1));
            ct = ct+1;
         end
      end
   end
   done_segs = done_segs(1:ct);
   %-------------------------
   %-------------------------
   % Write final file
   num = length(done_segs);
   for j = 1:num
      fldpln_model_write_final_file(done_segs(j),dr0,fldmn,fldmx,dh)
   end
   return;
end
