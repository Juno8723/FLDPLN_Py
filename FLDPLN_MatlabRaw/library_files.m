%--------------------------------
function library_files()
%--------------------------------
% reg = 'cks' or 'eks_lidar' or 'eks_lidar\neks_2020'
%=============================
% Enter home directory
%-----------------------------
dr0 = 'C:\Users\jsong33\Desktop\Research\FIM\FLDPLN\StudyAreas\Amite\Amite_UTM\';
%-----------------------------

%-----------------------------
% SLIE stream segments
shpf = [dr0,'vector\h15dh1_UTM_Enforced\Dissolve.shp'];
try
   seg_list = sort(readshapefield(shpf,{'GRID_CODE'}));
catch
   seg_list = sort(readshapefield(shpf,{'grid_code'}));
end
seg_tot = seg_list;
%seg_tot = setxor(seg_tot,1);
%seg_tot = 1;
%seg_tot = [1:12,17];
%-----------------------------
% Initialize flood parameters & the I/O file names
%-----------------------------
segdr = [dr0,'segment_files\h15dh1_UTM_Enforced\'];
dr = dir(segdr);
ldr = length(dr);
j = 3;
flg = 0;
while(~flg)
   nm0 = dr(j).name;
   if(nm0(1)=='h')
      f = find(nm0=='_');
      if(~isempty(f))
         hs = nm0(2:f(1)-1);
         dhs = nm0(f(1)+3:f(2)-1);
         break;
      end
   end
   j = j+1;
end
%hs = '15';
%dhs = '1';

reff = [dr0,'bil\Amite_Fdr.bil'];
fspf = [dr0,'mat\h15dh1_UTM_Enforced\fsp_info.mat'];
%----------------------
load(fspf,'fsp_x','fsp_y','segid');
pth_map = [segid fsp_x fsp_y];

%----------------------
% Convert SLIE files to contain map coordinates rather than vector IDs
%----------------------
segdr = [dr0,'segment_files\h15dh1_UTM_Enforced\'];
outdr = [dr0,'library_segment_files\h15dh1_UTM_Enforced\'];
if(~exist(outdr,'dir'))
   mkdir(outdr);
end
num = length(seg_tot);

header = {'FSP x','FSP y','floodplain pixel x','floodplain pixel y','DTF','DTF + fill depth'};
for j = 1:num
   js = int2str(seg_tot(j));
   outf = [outdr,'library_segment',js,'.mat'];
   if(~exist(outf,'file'))
      load([segdr,'h',hs,'_dh',dhs,'_seg',js],'fldpln');
      % fldpln:  FSP, floodplain pixel, DTF, DTF+fill

      pth_map0 = pth_map(pth_map(:,1)==seg_tot(j),2:3);
      npx = size(pth_map0,1);

      %=============================
      % Read spatial parameters from reference file
      %-------------------------
      refinfo = readbilheader(reff);
      c = refinfo.c; % cols
      ulx = refinfo.ulx; % ULX coordinate
      uly = refinfo.uly; % ULY coordinate
      pxsz = refinfo.pxszx; % pixel size
      %=============================

      fsp_map = zeros(size(fldpln,1),2);
      r0 = ceil(fldpln(:,1)/c)-1;
      c0 = fldpln(:,1)-r0*c-1;
      fsp_map(:,1) = c0*pxsz+ulx;
      fsp_map(:,2) = uly-r0*pxsz;

      fldpln0 = zeros(size(fldpln));
      ct = 0;
      for k = 1:npx
         fldpln1 = sortrows(fldpln((fsp_map(:,1)==pth_map0(k,1))+(fsp_map(:,2)==pth_map0(k,2))==2,:),3);
         npx0 = size(fldpln1,1);
         fldpln0(ct+(1:npx0),:) = fldpln1;
         ct = ct+npx0;
      end
      
      r0 = ceil(fldpln0(:,1)/c)-1;
      c0 = fldpln0(:,1)-r0*c-1;
      fsp_map(:,1) = c0*pxsz+ulx;
      fsp_map(:,2) = uly-r0*pxsz;

      fld_map = zeros(size(fldpln0,1),2);
      r0 = ceil(fldpln0(:,2)/c)-1;
      c0 = fldpln0(:,2)-r0*c-1;
      fld_map(:,1) = c0*pxsz+ulx;
      fld_map(:,2) = uly-r0*pxsz;
      
      fldpln = [fsp_map,fld_map,fldpln0(:,3:4)/0.3048];
      if(~all(ismember(fldpln(:,1:2),pth_map0,'rows')))
         size(fldpln,1)
         sum(~ismember(fldpln(:,1:2),pth_map0,'rows'))
%         f = find(~ismember(fldpln(:,1:2),pth_map(:,1:2),'rows'))
%         fldpln(f+(-1:1),:)'
%         fldpln0(f+(-1:1),:)'
%         return;
         error(['FSP mismatch for segment ',js]);
      end
      save(outf,'fldpln','header','-v7.3');
      disp(sprintf('%d of %d segments completed',j,num));
   %   if(j==19)
   %      fldpln = fldpln(fldpln(:,5)<=10,3:5);
   %      delete([outf(1:end-3),'xlsx'])
   %      xlswrite([outf(1:end-3),'xlsx'],{'X','Y','DTF'},'Sheet1','A1');
   %      xlswrite([outf(1:end-3),'xlsx'],fldpln,'Sheet1','A2');
   %      return;
   %   end
   end
end
%----------------------
