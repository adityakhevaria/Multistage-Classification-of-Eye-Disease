function generateDiagnosisME_Reporthealthy(img, Segmented_Image, ME_DiseaseYPred, ME_TreatmentOptions, ME_LifestyleRecommendations, ME_Metrics)

% Ensure DiseaseYPred, TreatmentOptions, and LifestyleRecommendations are correctly converted
ME_DiseaseYPred = char(ME_DiseaseYPred);  
ME_TreatmentOptions = char(ME_TreatmentOptions);  
ME_LifestyleRecommendations = char(ME_LifestyleRecommendations);  
currentDateTime = datestr(now); % Current date and time

% Generate PDF Report
pdfFileName = 'DiagnosisReport_Macular_Edema.pdf';
import mlreportgen.dom.*;
d = Document(pdfFileName, 'pdf');
open(d);

% Add Hospital Name centered on the first page
hospitalName = Paragraph('EYE CARE HOSPITAL MACULAR EDEMA REPORT', 'Heading1'); 
hospitalName.Style = {Color('red'), FontSize('18pt'), Bold(true)};  
append(d, hospitalName);
append(d, Paragraph(' ', 'Normal')); % Add space for better readability

% Add input and output images side by side
append(d, Paragraph('Input and Segmented Images:', 'Heading2'));  

% Input image
inputImgFileName = 'input_image.png';
imwrite(img, inputImgFileName); % Save input image
imgInput = Image(inputImgFileName);
imgInput.Style = {ScaleToFit(true)};  
imgInput.Width = '3in';  

% Output image
outputImgFileName = 'segmented_image.png';
imwrite(Segmented_Image, outputImgFileName); % Save segmented image
imgOutput = Image(outputImgFileName);
imgOutput.Style = {ScaleToFit(true)};  
imgOutput.Width = '3in';  

% Create a table for side-by-side images
imageTable = Table({imgInput, imgOutput});
imageTable.Style = {Border('solid', 'black', '1pt')};  
imageTable.Width = '6in'; 
append(d, imageTable);  

% Add space for better readability
append(d, Paragraph(' ', 'Normal'));

% Prepare the patient details and diagnosis information
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

patientDetailsTable = Table(patientDetailsData);
patientDetailsTable.Style = {Border('solid', 'black', '1pt'), FontSize('12pt'), Color('green')}; 
append(d, patientDetailsTable);  

% Add space for better readability
append(d, Paragraph(' ', 'Normal'));  

% Add metrics to the report
% Creating a new page for performance metrics
append(d, PageBreak); % Adds a page break to ensure metrics appear on the second page

% Add Heading for performance metrics
append(d, Paragraph('Performance Metrics:', 'Heading2'));  

% Organize the metrics data in a table
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

% Create a table for metrics
metricsTable = Table(metricsData);
metricsTable.Style = {Border('solid', 'black', '1pt'), FontSize('12pt'), Color('blue')}; 
append(d, metricsTable);  

% Add space for better readability
append(d, Paragraph(' ', 'Normal'));  

% Add closing remarks
append(d, Paragraph('Please review the attached report and let me know if you have any questions or require further information.', 'Normal'));
append(d, Paragraph(' ', 'Normal')); % Space before closing
append(d, Paragraph('Best regards,', 'Normal')); 
append(d, Paragraph(' ', 'Normal')); 
append(d, Paragraph('Your Healthcare Provider', 'NormalBold'));  

% Close the PDF document
close(d);

end
