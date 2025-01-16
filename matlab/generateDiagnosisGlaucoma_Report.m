function  generateDiagnosisGlaucoma_Report(Glucoma_img, Glucoma_Segmented_Image, Glucoma_colorImageOverlay, Glucoma_DiseaseYPred, Glucoma_TreatmentOptions, Glucoma_LifestyleRecommendations, Glucoma_Metrics, Glucoma_StageMetrics)

% Ensure inputs are in the correct format
    Glucoma_DiseaseYPred = char(Glucoma_DiseaseYPred);  
    Glucoma_TreatmentOptions = char(Glucoma_TreatmentOptions);  
    Glucoma_LifestyleRecommendations = char(Glucoma_LifestyleRecommendations);  
    currentDateTime = datestr(now); 

    % Generate PDF Report
    pdfFileName = 'DiagnosisReport_GLAUCOMA.pdf';
    import mlreportgen.dom.*;
    d = Document(pdfFileName, 'pdf');
    open(d);

    % ------------------------------- Page 1: Display Images -------------------------------

    % Add Hospital Name centered (Page 1)
    hospitalName = Paragraph('EYE CARE HOSPITAL  GLAUCOMA REPORT');
    hospitalName.Style = {Color('red'), FontSize('18pt'), Bold(true), HAlign('center')};
    append(d, hospitalName);
    append(d, Paragraph(' ')); % Add space for better readability

    % Add heading for images
    append(d, Paragraph('Input, Segmented, and Overlay Images:', 'Heading2'));

    % Save and add Input image
    inputImgFileName = 'input_image.png';
    imwrite(Glucoma_img, inputImgFileName); 
    imgInput = Image(inputImgFileName);
    imgInput.Style = {ScaleToFit(true)};
    imgInput.Width = '2.5in';

    % Save and add Segmented image
    segmentedImgFileName = 'segmented_image.png';
    imwrite(Glucoma_Segmented_Image, segmentedImgFileName); 
    imgSegmented = Image(segmentedImgFileName);
    imgSegmented.Style = {ScaleToFit(true)};
    imgSegmented.Width = '2.5in';

    % Save and add Overlay image
    overlayImgFileName = 'overlay_image.png';
    imwrite(Glucoma_colorImageOverlay, overlayImgFileName); 
    imgOverlay = Image(overlayImgFileName);
    imgOverlay.Style = {ScaleToFit(true)};
    imgOverlay.Width = '3in';

    % Create a table for side-by-side images (Input and Segmented Images)
    imageTable1 = Table({imgInput, imgSegmented}); 
    imageTable1.Style = {Border('solid', 'black', '1pt'), HAlign('left')};  % Visible borders for images
    append(d, imageTable1);

    % Add space and center the third image (Overlay Image)
    append(d, Paragraph(' ')); % Space between rows
    append(d, Paragraph('Overlay Image:', 'Heading3')); 
    overlayParagraph = Paragraph();
    overlayParagraph.Style = {HAlign('center')};  % Center the Overlay image
    append(overlayParagraph, imgOverlay);
    append(d, overlayParagraph);

    % Insert page break to move to the second page
    append(d, PageBreak);

    % ------------------------------- Page 2: Display Table and Metrics -------------------------------

    % Patient details and diagnosis information (Page 2)
    patientName = '***';
    patientAge = '***';
    patientGender = '***';
    patientID = '***';
    examinationDate = currentDateTime;

    % Create a table for patient details and diagnosis summary
    patientDetailsData = {
        'Subject:', 'Diagnosis Report'; 
        'Patient Name:', patientName; 
        'Age:', patientAge; 
        'Gender:', patientGender; 
        'Patient ID:', patientID; 
        'Date of Examination:', examinationDate;
        'Diagnosis:', Glucoma_DiseaseYPred; 
        'Treatment Options:', Glucoma_TreatmentOptions; 
        'Lifestyle Recommendations:', Glucoma_LifestyleRecommendations
    };

    % Create the table with visible borders
    patientDetailsTable = Table(patientDetailsData);
    patientDetailsTable.Style = {Border('solid', 'black', '1pt'), FontSize('12pt'), Color('green')};
    append(d, patientDetailsTable);

    % Add space for better readability
    append(d, Paragraph(' '));

    % ------------------------------- Metrics Section (Page 2) -------------------------------

    % Add general performance metrics (Page 2)
    append(d, Paragraph('Performance Metrics:', 'Heading2'));
    
    % General metrics table
    metricsData = {
        'Accuracy:', Glucoma_Metrics.Accuracy; 
        'PSNR:', Glucoma_Metrics.PSNR; 
        'Entropy:', Glucoma_Metrics.Entropy; 
        'AUC:', Glucoma_Metrics.AUC; 
        'Precision:', Glucoma_Metrics.Precision; 
        'Recall:', Glucoma_Metrics.Recall; 
        'F1 Score:', Glucoma_Metrics.F1_Score; 
        'Specificity:', Glucoma_Metrics.Specificity; 
        'Sensitivity:', Glucoma_Metrics.Sensitivity
    };

    % Create the metrics table with visible borders
    metricsTable = Table(metricsData);
    metricsTable.Style = {Border('solid', 'black', '1pt'), FontSize('12pt'), Color('blue')}; % Style for metrics
    append(d, metricsTable);  % Add the general metrics table

    % Add space for better readability
    append(d, Paragraph(' '));

    % Add stage-specific metrics
    append(d, Paragraph('Stage-specific Metrics:', 'Heading2'));

    % Stage-specific metrics table
    stageMetricsData = {
        'Stage Accuracy:', Glucoma_StageMetrics.Accuracy; 
        'Stage PSNR:', Glucoma_StageMetrics.PSNR; 
        'Stage Entropy:', Glucoma_StageMetrics.Entropy; 
        'Stage AUC:', Glucoma_StageMetrics.AUC; 
        'Stage Precision:', Glucoma_StageMetrics.Precision; 
        'Stage Recall:', Glucoma_StageMetrics.Recall; 
        'Stage F1 Score:', Glucoma_StageMetrics.F1_Score; 
        'Stage Specificity:', Glucoma_StageMetrics.Specificity; 
        'Stage Sensitivity:', Glucoma_StageMetrics.Sensitivity
    };

    % Create the stage metrics table with visible borders
    stageMetricsTable = Table(stageMetricsData);
    stageMetricsTable.Style = {Border('solid', 'black', '1pt'), FontSize('12pt'), Color('purple')}; % Style for stage metrics
    append(d, stageMetricsTable);  % Add the stage-specific metrics table

    % Add space for better readability
    append(d, Paragraph(' '));

    % Add closing remarks (Page 2)
    append(d, Paragraph('Please review the attached report and let me know if you have any questions or require further information.', 'Normal'));
    append(d, Paragraph(' '));
    append(d, Paragraph('Best regards,', 'Normal'));
    append(d, Paragraph(' '));
    append(d, Paragraph('Your Healthcare Provider', 'NormalBold'));

    % Close the PDF document
    close(d);
end
