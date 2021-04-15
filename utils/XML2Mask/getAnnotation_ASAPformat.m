function ANN = getAnnotation_ASAPformat(xmlFile)
%%%% things needed to work for both Aperio SVS and Ventana TIF files %%%%
%%% Ventana file is null-terminated; needs to be removed to create DOM object
%%% roi_id and area are stored differently

% debug
% xmlFile = 'C:\Users\anb12\Desktop\asdf.xml';
%%%%%%%

try
    xDoc = xmlread(xmlFile);
catch e
    if ~isempty(strfind(e.getReport(), 'Content is not allowed in trailing section.'))%????
        % assuming this errored due to null termination; so, import manually, remove null termination, and create DOM
        xmlString = unicode2native(fileread(xmlFile));
        xmlString = char(xmlString(1:2:end-1)); % remove null character at end
        xmlString = java.lang.String(xmlString);
        xmlStream = java.io.StringBufferInputStream(xmlString);
        factory = javaMethod('newInstance','javax.xml.parsers.DocumentBuilderFactory');
        builder = factory.newDocumentBuilder;
        
        xDoc = builder.parse(xmlStream);
        clear xmlString xmlStream factory builder;
    else
        rethrow(e);
    end
end

% get annotation layers
Annotations = xDoc.getElementsByTagName('Annotation');
ooo=Annotations.getLength-1;%?????
% traverse each region
for j=0:Annotations.getLength-1
    
    this_Annotation = Annotations.item(j);
    color = string(this_Annotation.getAttribute('Color'));
    red = '#ff0000';
    red2 = '#ff5500';
    yellow = '#F4FA58';
    green = '#005500';
    blue = '#0000ff';
    assert((color==red)|| (color==yellow)||(color==green)||(color==blue)||(color==red2));
    if color==yellow
        continue;
    end
    
    % get line color (assuming MS Access format, convert to RGB)
    % TODO: make this work with other formats as needed
    % get regions from this layer
    Regions = this_Annotation.getElementsByTagName('Coordinates');
    
    % traverse each region
    REGIONS=[];
    III=Regions.getLength-1;
    for i=0:Regions.getLength-1
        this_Region = Regions.item(i);
        
        
        % find child containing vertices for current ROI
        % we are also looking for area information along the way (for ventana format)
      
        
        % get vertices for current ROI
        X=[]; Y=[];
        Coordinate = this_Region.getFirstChild;
        while ~isempty(Coordinate)
            if strcmpi(Coordinate.getNodeName,'Coordinate')%???
                x = str2double(Coordinate.getAttribute('X').toCharArray');
                y = str2double(Coordinate.getAttribute('Y').toCharArray');
                X = [X; x];
                Y = [Y; y];
            end
            
            Coordinate = Coordinate.getNextSibling;
        end
        
        % compile regions struct
       
        REGIONS(i+1).X = X;
        REGIONS(i+1).Y = Y;
        
    end
    
    % compile annotation struct
  
    ANN(j+1).regions = REGIONS;

end