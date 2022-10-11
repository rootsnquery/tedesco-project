%% Paths

% this defines the directory paths, which contain all the data used in
% this code
pathdata = '/local/data/ra3063/home/';
pathoutput = '/local/data/ra3063/home/output/';
pathMODIS_MARproj_6km = '/local/data/ra3063/data/MOD09GA_MARproj_6_5km/';
pathMODISmask_MARproj_6km = '/local/data/ra3063/data/MOD10A1_mask_MARproj_6_5km/';
pathMODISalbedo_MARproj_6km = '/local/data/ra3063/data/MOD10A1_albedo_MARproj_6_5km/';
pathMAR_6km = '/local/data/ra3063/data/MAR/MARv3.12/6_5km/';

% this is a separate code that sets the projection, you don't need to know
% this code.
RafParam_6_5km

% the MAR output resolution is actually 6.5km, not 6km

%% MODIS variables

% This creates the latitude and longitude grid for the satellite data
[xMODIS, RxMAR] = geotiffread(fullfile(pathMODIS_MARproj_6km, 'lon.tif'));
[yMODIS, RyMAR] = geotiffread(fullfile(pathMODIS_MARproj_6km, 'lat.tif'));
[xMAR, yMAR] = geo2MAR(xMODIS, yMODIS);

% amount of days in June, July and August (JJA)
nfiles = 92;

% to be used later in the loop to call files
yrname = ['00'; '01'; '02'; '03'; '04'; '05'; '06'; '07'; '08'; '09'; ...
          '10'; '11'; '12'; '13'; '14'; '15'; '16'; '17'; '18'; '19'; '20'; '21'];

% create latitude filter to only include cells below 70 degrees north
lat70 = yMODIS;
lat70(lat70 > 70) = 0;
lat70(lat70 ~= 0) = 1;


%% MAR variables

% load latitude and longitude from MAR
lat_ERA = ncread(fullfile(pathMAR_6km, 'MARv3.12-6_5km-daily-ERA5-2020.nc'), 'LAT');
lon_ERA = ncread(fullfile(pathMAR_6km, 'MARv3.12-6_5km-daily-ERA5-2020.nc'), 'LON');

% slightly adjust MAR latitude and longitude to fit with MODIS latitude and
% longitude
[x_ERA, y_ERA] = geo2MAR(lon_ERA, lat_ERA);
x_ERA = x_ERA -0.5;
y_ERA = y_ERA - 0.5;
[lat_ERA, lon_ERA] = MAR2geo(x_ERA,y_ERA);

% load variables mask (MSK), soil type (SOL), and surface height (SH)
MSK_ERA = ncread(fullfile(pathMAR_6km, 'MARv3.12-6_5km-daily-ERA5-2020.nc'), 'MSK');
SOL_ERA = ncread(fullfile(pathMAR_6km, 'MARv3.12-6_5km-daily-ERA5-2020.nc'), 'SOL');
SH_ERA = ncread(fullfile(pathMAR_6km, 'MARv3.12-6_5km-daily-ERA5-2020.nc'), 'SH');

% create equilibrium line altitude (ELA) cutoff at 1679m. we assume that
% ice will not be present above this altitude so we don't include it in the
% analysis
SMBH_095 = 1679;
SH_ERA(SH_ERA > SMBH_095) = 0; SH_ERA(SH_ERA ~= 0) = 1;

% group and label islands. 8 for inclusion of diagonal cells
gland_ERA = bwlabel(SOL_ERA, 8);
% only use island 'greenland mainland' with value 1
gland_ERA(gland_ERA~=1) = 0;
% ice sheet area is greenland mainland multiplied by ice sheet area mask
gice_ERA = gland_ERA .* MSK_ERA/100;
giceELA_ERA = gland_ERA .* MSK_ERA/100 .* SH_ERA;
% resolution in km^2
res_ERA = 6.5^2;
% total area of greenland ice sheet in km^2
garea_ERA = sum(sum(gice_ERA)) * res_ERA;


%% MAR BIE

% total years in analysis
yrs = 22;

% 22 years, from 2000 to 2021. 92 days, from June 1 to August 31
s_ERA_6km = zeros(230, 415, yrs, 92);
al_ERA_6km = zeros(230, 415, yrs, 92);

% create 4D matrix (s) in size of MAR with 6.5km resolution for each day
% and each year. a is for albedo
s = zeros(415, 230, yrs, 92);
a = zeros(415, 230, yrs, 92);

% create matrices for albedo calculations
aMOD_BIE_day_6km = zeros(415, 230, 19, 92);
aMOD_BIEavets_day_6km = zeros(yrs, 92);
aERA_BIE_day_6km = zeros(415, 230, 19, 92);
aERA_BIEavets_day_6km = zeros(yrs, 92);

aMOD_BIEavets_day_6km_lat70 = zeros(yrs, 92);
aERA_BIEavets_day_6km_lat70 = zeros(yrs, 92);

sboth_6km = zeros(415, 230, yrs, 92);
sboth_6km_lat70 = zeros(415, 230, yrs, 92);
sboth_sum_6km = zeros(yrs, 92);
sboth_sum_6km_lat70 = zeros(yrs, 92);


% loop over years from 2000 to 2021 with t=1 is 2000 and t=22 is 2021. it
% is stored in biMOD(t,:), so biMOD(actual year+1,:)
for t = 1:yrs
    
    disp('year='); disp(t)
    
    %%%%%
    % MAR
    %%%%%
    
    % create year-variable to be used in calling matrix elements
    yearmat = yrs + 1999;
    
    % create a number for june 1st
    june1 = datenum(yearmat,6,1) - datenum(yearmat,1,1) + 1;
    % create a number for august 31st
    aug31 = datenum(yearmat,8,31) - datenum(yearmat,1,1) + 1;
    % create new variable to use in string
    year1 = yrs - 1;
    
    % read SHSN2, RO1 and AL2 to compute bare ice extent for each year and day
    % SHSN2 = snow pack height above ice. 0 for no snow or only snow. only
    % gives a value if top 20m has snow and ice
    % RO1 = snow/firn/ice density. 0 for land or ocean
    % AL2 = albedo
    if year1 < 10
        SHSN2 = ncread(fullfile(pathMAR_6km, sprintf('MARv3.12-6_5km-daily-ERA5-200%d.nc', year1)), 'SHSN2');
        RO1 = ncread(fullfile(pathMAR_6km, sprintf('MARv3.12-6_5km-daily-ERA5-200%d.nc', year1)), 'RO1');
        al_ERAdum = ncread(fullfile(pathMAR_6km, sprintf('MARv3.12-6_5km-daily-ERA5-200%d.nc', year1)), 'AL2');
    else
        SHSN2 = ncread(fullfile(pathMAR_6km, sprintf('MARv3.12-6_5km-daily-ERA5-20%d.nc', year1)), 'SHSN2');
        RO1 = ncread(fullfile(pathMAR_6km, sprintf('MARv3.12-6_5km-daily-ERA5-20%d.nc', year1)), 'RO1');
        al_ERAdum = ncread(fullfile(pathMAR_6km, sprintf('MARv3.12-6_5km-daily-ERA5-20%d.nc', year1)), 'AL2');
    end
    
    % loop through all days (june1-aug31) for each year and compute which
    % cells are bare ice. also compute corresponding albedo value for these
    % cells
    for day = june1:aug31
        
        % use daymat to call the right element in the matrices
        daymat = day - june1 + 1;
        
        SHSN2day = SHSN2(:,:,1,day);
        % change cells with no snow or only snow into -1's
        SHSN2day(SHSN2day==0) = -1;
        % change cells with snow and ice into 0's --> so don't use these
        SHSN2day(SHSN2day>-1) = 0;
        % change -1's into 1's --> so do use these
        SHSN2day(SHSN2day==-1) = 1;
        
        % use top 10 surface layers (10 out of 18) --> top 1m
        RO1day = mean(RO1(:,:,1:10,day), 3);
        % don't use cells with density lower than 907 kg/m3. Xavier and Patrick, personal comm.
        RO1day(RO1day<907) = 0;
        % use all other cells
        RO1day(RO1day~=0) = 1;
        
        % compute which cells have only ice, that also have an average
        % density in the top 1m of more than 907 kg/m3, and that are also
        % in the ice mask create above
        s_ERA_6km(:,:,t,daymat) = RO1day .* SHSN2day .* giceELA_ERA;
        
    end
    
    % reduce albedo matrix to only include june1-aug31
    al_ERA_6km(:,:,t,:) = al_ERAdum(:,:,1,june1:aug31);
    
    
    
    %%%%%%%
    % MODIS
    %%%%%%%
    
    % create list of names of all .tif files for current year
    name = convertCharsToStrings(strcat('20', yrname(t,:), '*.tif'));
    name = convertStringsToChars(name);
    filelist = dir([pathMODIS_MARproj_6km, name]);
    filelistmask = dir([pathMODISmask_MARproj_6km, name]);
    filelistalbedo = dir([pathMODISalbedo_MARproj_6km, name]);

    % loop over each day in JJA in current year
    for i = 1 : nfiles
        
        imat = i;
        
        if rem(i, 20) == 0
            disp(i)
        end
        
        % data clean up. i.e. remove some days with bad data
        if t == 1
            if i == 12 || i == 27
                continue
            end
        end
        if t == 1 && i >= 67 && i <= 79 % 80 files in total
            continue
        end
        if t == 1 && i > 79
            imat = i - 13;
        end
        
        if t == 2 && i == 15
            continue
        end
        if t == 2 && i >= 16 && i <= 32 % 75 files in total
            continue
        end
        if t == 2 && i > 32
            imat = i - 17;
        end
        
        if t == 4
            if i == 18 || i == 59 || i == 61 || i == 67
                continue
            end
        end

        if t == 7
            if i == 20 || i == 21 || i == 84
                continue
            end
        end
        
        if t == 10
            if i == 87
                continue
            end
        end
        
        if t == 11
            if i == 72
                continue
            end
        end
        
        if t == 13
            if i == 6
                continue
            end
        end
        
        
        % read bare ice extent (BIE) tif file for current day and year and create dummy     
        sdum = double(geotiffread([pathMODIS_MARproj_6km, filelist(imat).name]));
        
        % read cloud mask tif file for current day and year and keep only
        % clouds as 1's, the rest as 0's
        smask = double(geotiffread([pathMODISmask_MARproj_6km, filelistmask(imat).name]));
        smask(smask >= 3 | smask < 0) = 0;
                
        % subtract cloud mask from BIE (this is a second subtraction.
        % the first was performed before the projection transformation and
        % resampling by Shujie, which results in possible missed cells)
        s(:,:,t,i) = sdum - smask;
        
        % read albedo tif file for current day and year and create dummy     
        a(:,:,t,i) = double(geotiffread([pathMODISalbedo_MARproj_6km, filelistalbedo(imat).name]));
        
    end
    
    % transform everything that is not ice into 0's. clouds are -1, bare
    % ice is 1
    s(s == -1 | s == 2 | s == 3) = 0;

    % apply ELA height filter, i.e. cells above 1679m cannot be bare ice
    s = s .* flipud(transpose(SH_ERA));

    % create new matrix to calculate sum of bare ice pixels
    sbi_6km = s;
    
    % only take into account those cells which are ice for more than 50% of
    % the cell
    sbi_6km(sbi_6km < 0.5) = 0;
    s_ERA_6km(s_ERA_6km < 0.5) = 0;
    
    for is = 1:nfiles
        
        if rem(is, 20) == 0
            disp('is='); disp(is)
        end
        
        % determine which cells are bare ice for both MODIS and MAR
        sboth_6km(:,:,t,is) = sbi_6km(:,:,t,is) .* flipud(transpose(s_ERA_6km(:,:,t,is)));
        sboth_6km(sboth_6km == 0) = NaN;
        sboth_sum_6km(t,is) = sum(sum(~isnan(sboth_6km(:,:,t,is))));
        
        % determine which cells are bare ice for both MODIS and MAR below
        % 70 N
        sboth_6km_lat70(:,:,t,is) = sbi_6km(:,:,t,is) .* flipud(transpose(s_ERA_6km(:,:,t,is))) .* lat70;
        sboth_6km_lat70(sboth_6km_lat70 == 0) = NaN;
        sboth_sum_6km_lat70(t,is) = sum(sum(~isnan(sboth_6km_lat70(:,:,t,is))));
        
    end
end


%%%%%%%%
% Albedo
%%%%%%%%

% turn albedo values lower than 0 and higher than 100 into NaN
a(a < 0 | a > 100) = NaN;

% albedo values in MAR are between 0 and 1. albedo values in MODIS are
% between 0 and 100, that's why I divide by 100 in some of the function
% below

for t = 1:yrs
    
    for ia = 1:nfiles
        
        if sboth_sum_6km(t,ia) > 0
            
            % find MODIS albedo values for cells which are bare ice for
            % both MODIS and MAR
            aMOD_BIE_day_6km(:,:,t,ia) = a(:,:,t,ia) .* sboth_6km(:,:,t,ia);
            
            % create daily averages of ice albedo in MODIS
            aMOD_BIEavets_day_6km(t,ia) = (nansum(nansum(aMOD_BIE_day_6km(:,:,t,ia), 1), 2) / ...
                                           nansum(nansum(sboth_6km(:,:,t,ia), 1), 2)) / 100;
            % create daily averages of ice albedo in MODIS below 70 N
            aMOD_BIEavets_day_6km_lat70(t,ia) = (nansum(nansum(aMOD_BIE_day_6km(:,:,t,ia) .* lat70, 1), 2) / ...
                                           nansum(nansum(sboth_6km_lat70(:,:,t,ia), 1), 2)) / 100;
                                           
            % find MAR albedo values for cells which are bare ice for
            % both MODIS and MAR
            aERA_BIE_day_6km(:,:,t,ia) = flipud(transpose(al_ERA_6km(:,:,t,ia))).* sboth_6km(:,:,t,ia);

            % create daily averages of ice albedo in MODIS
            aERA_BIEavets_day_6km(t,ia) = nansum(nansum(aERA_BIE_day_6km(:,:,t,ia), 1), 2) / ...
                                           nansum(nansum( sboth_6km(:,:,t,ia), 1), 2);
            
            % create daily averages of ice albedo in MODIS below 70 N
            aERA_BIEavets_day_6km_lat70(t,ia) = nansum(nansum(aERA_BIE_day_6km(:,:,t,ia) .* lat70, 1), 2) / ...
                                           nansum(nansum(sboth_6km_lat70(:,:,t,ia), 1), 2);
                                       
        end
        
    end
    
end

% turn 0's into NaN
aMOD_BIEavets_day_6km(aMOD_BIEavets_day_6km == 0) = NaN;
aERA_BIEavets_day_6km(aERA_BIEavets_day_6km == 0) = NaN;

aMOD_BIEavets_day_6km_lat70(aMOD_BIEavets_day_6km_lat70 == 0) = NaN;
aERA_BIEavets_day_6km_lat70(aERA_BIEavets_day_6km_lat70 == 0) = NaN;


%% Save outputs

save(fullfile(pathoutput, 'aMOD_BIEavets_day_6km.mat'), 'aMOD_BIEavets_day_6km')
save(fullfile(pathoutput, 'aERA_BIEavets_day_6km.mat'), 'aERA_BIEavets_day_6km')
save(fullfile(pathoutput, 'sboth_sum_6km.mat'), 'sboth_sum_6km')

save(fullfile(pathoutput, 'aMOD_BIEavets_day_6km_lat70.mat'), 'aMOD_BIEavets_day_6km_lat70')
save(fullfile(pathoutput, 'aERA_BIEavets_day_6km_lat70.mat'), 'aERA_BIEavets_day_6km_lat70')
save(fullfile(pathoutput, 'sboth_sum_6km_lat70.mat'), 'sboth_sum_6km_lat70')


s_6km = s;
a_6km = a;

save(fullfile(pathoutput, 's_6km.mat'), 's_6km')
save(fullfile(pathoutput, 'a_6km.mat'), 'a_6km')








