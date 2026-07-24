%------------------------------
%function [mn,mx,bg] = scantif(fnam)
%------------------------------
% This function scans ESRI BIL files to extract the minimum,
% maximum, and NoData values.
%------------------------------
function [mn,mx,bg] = scanbil(fnam)
%------------------------------

filinfo = readbilheader(fnam);
rows = filinfo.r;
cols = filinfo.c;
datfmt = filinfo.fmt;
bytord = filinfo.bytord;
bg = filinfo.nodata;

dv = floor(rows/100);
md = rows-dv*100;

if(ismember(datfmt,{'single','double'}))
   mn = realmax(datfmt);
   mx = -realmax(datfmt);
elseif(~isempty(strfind(datfmt,'int')))
   mn = intmax(datfmt);
   mx = intmin(datfmt);
end
mn2 = mn;
mx2 = mx;

if(bytord==1)
   fid = fopen(fnam,'rb','b');
else
   fid = fopen(fnam,'rb');
end

smbg = 0;
for j = 1:dv
   dat = eval([datfmt,'(fread(fid,cols*100,datfmt));']);
   mn = min(mn,min(dat));
   if(any(dat>mn))
      mn2 = min(mn2,min(dat(dat>mn)));
   end
   mx = max(mx,max(dat));
   if(any(dat<mx))
      mx2 = max(mx2,max(dat(dat<mx)));
   end
   if(any(dat==bg))
      smbg = smbg+sum(dat==bg);
   end
end
if(md)
   dat = eval([datfmt,'(fread(fid,md,datfmt));']);
   mn = min(mn,min(dat));
   if(any(dat>mn))
      mn2 = min(mn2,min(dat(dat>mn)));
   end
   mx = max(mx,max(dat));
   if(any(dat<mx))
      mx2 = max(mx2,max(dat(dat<mx2)));
   end
   if(any(dat==bg))
      smbg = smbg+sum(dat==bg);
   end
end
fclose(fid);

if(and(~smbg,~ismember(datfmt,{'uint8','int8','uint16','int16'})))
   if(or(abs(mn2-mn)>1,abs(mx2-mx)>1))
      if(abs(mn2-mn)>=abs(mx2-mx))
         bg = mn; mn = mn2;
      else
         bg = mx; mx = mx2;
      end
   else
      bg = NaN;
   end
elseif(bg==mx)
   mx = mx2;
elseif(bg==mn)
   mn = mn2;
end
