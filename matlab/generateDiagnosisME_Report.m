function  generateDiagnosisME_Report(img, Segmented_Image, ME_colorImageOverlay, ME_DiseaseYPred, ME_TreatmentOptions, ME_LifestyleRecommendations, ME_Metrics, ME_StageMetrics)
    
% Ensure inputs are in the correct format
    ME_DiseaseYPred = char(ME_DiseaseYPred);  
    ME_TreatmentOptions = char(ME_TreatmentOptions);  
    ME_LifestyleRecommendations = char(ME_LifestyleRecommendations);  
    currentDateTime = datestr(now); 

    % Generate PDF Report
    pdfFileName = 'DiagnosisReport_Macular_Edema.pdf';
    import mlreportgen.dom.*;
    d = Document(pdfFileName, 'pdf');
    open(d);

    % ------------------------------- Page 1: Display Images -------------------------------

    % Add Hospital Name centered (Page 1)
    hospitalName = Paragraph('EYE CARE HOSPITAL MACULAR EDEMA  REPORT');
    hospitalName.Style = {Color('red'), FontSize('18pt'), Bold(true), HAlign('center')};
    append(d, hospitalName);
    append(d, Paragraph(' ')); % Add space for better readability

    % Add heading for images
    append(d, Paragraph('Input, Segmented, and Overlay Images:', 'Heading2'));

    % Save and add Input image
    inputImgFileName = 'input_image.png';
    imwrite(img, inputImgFileName); 
    imgInput = Image(inputImgFileName);
    imgInput.Style = {ScaleToFit(true)};
    imgInput.Width = '2.5in';

    % Save and add Segmented image
    segmentedImgFileName = 'segmented_image.png';
    imwrite(Segmented_Image, segmentedImgFileName); 
    imgSegmented = Image(segmentedImgFileName);
    imgSegmented.Style = {ScaleToFit(true)};
    imgSegmented.Width = '2.5in';

    % Save and add Overlay image
    overlayImgFileName = 'overlay_image.png';
    imwrite(ME_colorImageOverlay, overlayImgFileName); 
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
        'Diagnosis:', ME_DiseaseYPred; 
        'Treatment Options:', ME_TreatmentOptions; 
        'Lifestyle Recommendations:', ME_LifestyleRecommendations
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
        'Accuracy:', ME_Metrics.Accuracy; 
        'PSNR:', ME_Metrics.PSNR; 
        'Entropy:', ME_Metrics.Entropy; 
        'AUC:', ME_Metrics.AUC; 
        'Precision:', ME_Metrics.Precision; 
        'Recall:', ME_Metrics.Recall; 
        'F1 Score:', ME_Metrics.F1_Score; 
        'Specificity:', ME_Metrics.Specificity; 
        'Sensitivity:', ME_Metrics.Sensitivity
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
        'Stage Accuracy:', ME_StageMetrics.Accuracy; 
        'Stage PSNR:', ME_StageMetrics.PSNR; 
        'Stage Entropy:', ME_StageMetrics.Entropy; 
        'Stage AUC:', ME_StageMetrics.AUC; 
        'Stage Precision:', ME_StageMetrics.Precision; 
        'Stage Recall:', ME_StageMetrics.Recall; 
        'Stage F1 Score:', ME_StageMetrics.F1_Score; 
        'Stage Specificity:', ME_StageMetrics.Specificity; 
        'Stage Sensitivity:', ME_StageMetrics.Sensitivity
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
