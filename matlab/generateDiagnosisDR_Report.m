function generateDiagnosisDR_Report(img, Segmented_Image, colorImageOverlay, DiseaseYPred, TreatmentOptions, LifestyleRecommendations, Metrics, StageMetrics)
    % Ensure inputs are in the correct format
    DiseaseYPred = char(DiseaseYPred);  
    TreatmentOptions = char(TreatmentOptions);  
    LifestyleRecommendations = char(LifestyleRecommendations);  
    currentDateTime = datestr(now); 

    % Generate PDF Report
    pdfFileName = 'DiagnosisReport_Diabetic_Retinopathy.pdf';
    import mlreportgen.dom.*;
    d = Document(pdfFileName, 'pdf');
    open(d);

    % ------------------------------- Page 1: Display Images -------------------------------

    % Add Hospital Name centered (Page 1)
    hospitalName = Paragraph('EYE CARE HOSPITAL DIABETIC RETINOPATHY REPORT');
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
    imwrite(colorImageOverlay, overlayImgFileName); 
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
        'Diagnosis:', DiseaseYPred; 
        'Treatment Options:', TreatmentOptions; 
        'Lifestyle Recommendations:', LifestyleRecommendations
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
        'Accuracy:', Metrics.Accuracy; 
        'PSNR:', Metrics.PSNR; 
        'Entropy:', Metrics.Entropy; 
        'AUC:', Metrics.AUC; 
        'Precision:', Metrics.Precision; 
        'Recall:', Metrics.Recall; 
        'F1 Score:', Metrics.F1_Score; 
        'Specificity:', Metrics.Specificity; 
        'Sensitivity:', Metrics.Sensitivity
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
        'Stage Accuracy:', StageMetrics.Accuracy; 
        'Stage PSNR:', StageMetrics.PSNR; 
        'Stage Entropy:', StageMetrics.Entropy; 
        'Stage AUC:', StageMetrics.AUC; 
        'Stage Precision:', StageMetrics.Precision; 
        'Stage Recall:', StageMetrics.Recall; 
        'Stage F1 Score:', StageMetrics.F1_Score; 
        'Stage Specificity:', StageMetrics.Specificity; 
        'Stage Sensitivity:', StageMetrics.Sensitivity
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
