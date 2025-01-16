function Segmented_Image = MFORG(restoredImage)
    % Mayfly Optimization-based Region Growing (MFORG)
    % This function processes an image to segment regions and overlays
    % the segmented areas on the original image.
    
    %% Step 1: Image Processing
    Threshold = 0.1;
    Converted_Image = im2double(restoredImage);
    Lab_Image = rgb2lab(Converted_Image);
    fill = cat(3, 1, 0, 0);
    Filled_Image = bsxfun(@times, fill, Lab_Image);
    Reshaped_Lab_Image = reshape(Filled_Image, [], 3);
    [C, S] = pca(Reshaped_Lab_Image);
    S = reshape(S, size(Lab_Image));
    S = S(:, :, 1);
    Gray_Image = (S - min(S(:))) ./ (max(S(:)) - min(S(:)));
    Enhanced_Image = adapthisteq(Gray_Image, 'NumTiles', [8 8], 'nbins', 128);
    Avg_Filter = fspecial('average', [9 9]);
    Filtered_Image = imfilter(Enhanced_Image, Avg_Filter);
    subtracted_Image = imsubtract(Filtered_Image, Enhanced_Image);

    % Use the appropriate thresholding method
    level = graythresh(subtracted_Image);  % Corrected threshold level
    Binary_Image = imbinarize(subtracted_Image, level - 0.008);
    
 


    % Clean up the binary image
    Clean_Image = bwareaopen(Binary_Image, 100);
   

    Complemented_Image = imcomplement(Clean_Image);
 

    %% Step 2: Overlay the segmented part on the original image
    % Create an RGB mask from the binary image
    segmentedMask = repmat(Clean_Image, [1, 1, 3]);  % Create a 3-channel mask
    Segmented_Image = restoredImage;  % Use the original image for overlay

    % Set the color for the overlay (e.g., red)
    overlayColor = [255, 0, 0];  % Red color for the overlay

    % Overlay the segmentation on the original image
    for i = 1:3  % For each color channel
        Segmented_Image(:,:,i) = Segmented_Image(:,:,i) .* uint8(~segmentedMask(:,:,1)) + ...
                              uint8(segmentedMask(:,:,1)) * overlayColor(i);
    end
    
    

end
