%=============================
% function riverside_fsp_info_table_poudre()
%=============================
% equation for BkfD (ft) = 1.04*A^0.3136, where A is drainage area in sq miles
% put a multiplier in front of this equation
%
% OUTPUT:
%  bkfd_dtf = n-by-2 matrix with stream pixel ID in the first column and
%             BKFD-based DTF in the second
%
%=============================
function fsp_info_table()
%=============================
% reg = 'cks' or 'eks_lidar'
%=============================
% Enter home directory
%-----------------------------
dr0 = 'C:\Users\jsong33\Desktop\Research\FIM\FLDPLN\StudyAreas\Amite\Amite_UTM\';
strnm = {'Amite'};
targetname = 'Amite';

fdrf = [dr0,'bil\Amite_Fdr.bil'];
filf = [dr0,'bil\Amite_Fil.bil'];
facf = [dr0,'bil\Amite_Fac.bil'];
segf = [dr0,'bil\h15dh1_UTM_Enforced\str_segid.bil'];
matf = [dr0,'mat\h15dh1_UTM_Enforced\seg_info.mat'];
fspf = [dr0,'mat\h15dh1_UTM_Enforced\fsp_info.mat'];

if(~exist([dr0,'mat\h15dh1_UTM_Enforced\'],'dir'))
   mkdir([dr0,'mat\h15dh1_UTM_Enforced\']);
end
reff = [dr0,'bil\Amite_Fdr.bil'];
shpf = [dr0,'vector\h15dh1_UTM_Enforced\Dissolve.shp'];
try
   seg_list = sort(readshapefield(shpf,{'GRID_CODE'}));
catch
   seg_list = sort(readshapefield(shpf,{'grid_code'}));
end

%=============================
% Read spatial parameters from FIL and open it for reading
%-------------------------
filinfo = readbilheader(filf);
r = filinfo.r; % rows
c = filinfo.c; % cols
ulx = filinfo.ulx; % ULX centroid
uly = filinfo.uly; % ULY centroid
pxsz = filinfo.pxszx; % pixel size (assuming meter units!!!)
filbytord = filinfo.bytord; % byte order; if 1 then open file with 'b' option
filfmt = filinfo.fmt; % FIL BIL file data format
filbyt = filinfo.nbyts; % FIL BIL file number of bytes per data value
if(filbytord)
   ffil = fopen(filf,'rb','b');
else
   ffil = fopen(filf,'rb');
end
%=============================

%=============================
% Open FAC for reading
%-------------------------
facinfo = readbilheader(facf);
r0 = facinfo.r; % rows
c0 = facinfo.c; % cols
if(or(r~=r0,c~=c0))
   error('FIL and FAC do not have the same image size');
end
facbytord = facinfo.bytord; % byte order; if 1 then open file with 'b' option
facfmt = facinfo.fmt; % FAC BIL file data format
facbyt = facinfo.nbyts; % FAC BIL file number of bytes per data value
if(facbytord)
   ffac = fopen(facf,'rb','b');
else
   ffac = fopen(facf,'rb');
end
%=============================

%=============================
% Open STR_SEGID for reading
%-------------------------
if(1<0)
seginfo = readbilheader(segf);
r0 = seginfo.r; % rows
c0 = seginfo.c; % cols
if(or(r~=r0,c~=c0))
   error('FIL and STR_SEGID do not have the same image size');
end
segbytord = seginfo.bytord; % byte order; if 1 then open file with 'b' option
segfmt = seginfo.fmt; % STR_SEGID BIL file data format
segbyt = seginfo.nbyts; % STR_SEGID BIL file number of bytes per data value
if(segbytord)
   fseg = fopen(segf,'rb','b');
else
   fseg = fopen(segf,'rb');
end
end
%=============================

%=============================
% Initialize flood parameters & the I/O file names
%-----------------------------
nstr = length(strnm);
stream_lookup_table = cell(nstr,2);
stream_lookup_table(:,2) = strnm';
for j = 1:nstr
   stream_lookup_table{j,1} = j;
end
if(~exist([fspf(1:end-3),'xlsx'],'file'))
   xlswrite([fspf(1:end-3),'xlsx'],{'StrID','Stream Name'},'stream_lookup_table','A1');
   xlswrite([fspf(1:end-3),'xlsx'],stream_lookup_table,'stream_lookup_table','A2');
end

%----------------------
% Convert stream pixel BIL vector IDs to map coordinates
%----------------------
if(~exist(fspf,'file'))
   load(matf,'strpx');
   % strpx = [BIL pixel ID, FAC value, nat seg ID, refined seg ID]
   strpx = strpx(ismember(strpx(:,4),seg_list),:);
   pth = strpx(:,1);
   
   %=============================
   % Read spatial parameters from reference file
   %-------------------------
   refinfo = readbilheader(reff);
   c = refinfo.c; % cols
   ulx = refinfo.ulx; % ULX coordinate
   uly = refinfo.uly; % ULY coordinate
   pxsz = refinfo.pxszx; % pixel size
   %=============================

%   pth_map = [zeros(size(pth,1),2) pth(:,3)];
   pth_map = [zeros(size(pth,1),2) ones(size(pth,1),1)];
   r0 = ceil(pth(:,1)/c)-1;
   c0 = pth(:,1)-r0*c-1;
   pth_map(:,1) = c0*pxsz+ulx;
   pth_map(:,2) = uly-r0*pxsz;

   fsp_x = pth_map(:,1);
   fsp_y = pth_map(:,2);
   str_id = pth_map(:,3);
   %----------------------

   %=============================
   % Determine file coordinates (row & col) for each point
   % Integer values correspond with pixel centroids
   %-------------------------
   x0 = fsp_x; y0 = fsp_y;
   r0 = round((uly-y0)/pxsz);
   c0 = round((x0-ulx)/pxsz);
   nfsp = length(fsp_x);
   %=============================

   segid = strpx(:,4);
   fil = zeros(nfsp,1);
   fac = zeros(nfsp,1);

   for j = 1:nfsp
      fseek(ffil,filbyt*(r0(j)*c+c0(j)),-1);
      fil(j) = fread(ffil,1,filfmt);
      fseek(ffac,facbyt*(r0(j)*c+c0(j)),-1);
      fac(j) = fread(ffac,1,facfmt);
      if(~mod(j,5000))
         disp(sprintf('%d of %d FSPs completed',j,nfsp));
      end
   end
   fclose('all');
else
   load(fspf);
end
%=============================
tabl = [fsp_x fsp_y str_id segid fil fac];
fsp_x = tabl(:,1); fsp_y = tabl(:,2); str_id = tabl(:,3);
segid = tabl(:,4); fil = tabl(:,5); fac = tabl(:,6);
save(fspf,'fsp_x','fsp_y','str_id','segid','fil','fac');

% Convert FIL from meters to feet
tabl(:,5) = tabl(:,5)/0.3048;

[min(tabl(:,5)) max(tabl(:,5))]
%return

tablhdr = {'X','Y','StrID','SegID','FIL'};
xlswrite([fspf(1:end-3),'xlsx'],tablhdr,'FSP','A1');
xlswrite([fspf(1:end-3),'xlsx'],tabl(:,1:5),'FSP','A2');

if(1>0)
if(~exist([dr0,'ftp'],'dir'))
   mkdir([dr0,'ftp'])
end
%dos(['copy ',fspf(1:end-3),'xlsx',' ',dr0,'ftp\',strnm,'_fsp_info.xlsx']);
%Delete_sheets_Excel([dr0,'ftp\',strnm,'_fsp_info.xlsx'],'Sheet1');

if(~exist('seg_info','var'))
   load(matf,'seg_info');
end
n = size(seg_info,1);
seg_info = [(1:n)' seg_info(:,5:6)];
seghdr = {'segID','seg_len','segID_ds'};
xlswrite([dr0,'ftp\segment_network_info_',targetname,'.xlsx'],seghdr,'Sheet1','A1');
xlswrite([dr0,'ftp\segment_network_info_',targetname,'.xlsx'],seg_info,'Sheet1','A2');
end