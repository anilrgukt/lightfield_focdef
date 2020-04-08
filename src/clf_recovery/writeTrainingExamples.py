function createFlag = WriteTrainingExamples(inImgs, ref, refPos, outputDir, writeOrder, startInd, createFlag, arraySize)
global param
%firstBatch = WriteTrainingExamples(pInImgs, pRef, outputFolder, writeOrder, (ns-1) * numPatches * param.numRefs + 1, firstBatch, numTotalPatches);
chunkSize = 1000;
fileName = sprintf('%straining', outputDir);
numTotalPatches = arraySize;
numH5 = param.numH5;
sizeH5 = floor(numTotalPatches/numH5);
%arraySize = sizeH5;

[~, numElements] = size(refPos);

bins = 1:sizeH5:numTotalPatches;
bins = [bins numTotalPatches+1];

for k = 1 : numElements
    
    j = k + startInd - 1;
    
    curInImgs = inImgs(:, :, :, k);
    curRef = ref(:, :, :, k);
    
	w = ceil(writeOrder(j)/sizeH5);
	fName = sprintf('%s_%s.h5',fileName,num2str(w));

	arraySize = bins(w+1)-bins(w);
	startLoc = mod(writeOrder(j),sizeH5) + 1;
    SaveHDF(fName, '/data_tr', single(curInImgs), PadWithOne(size(curInImgs), 4), [1, 1, 1, startLoc], chunkSize, createFlag(w), arraySize);
    SaveHDF(fName, '/label_tr', single(curRef), PadWithOne(size(curRef), 4), [1, 1, 1, startLoc], chunkSize, createFlag(w), arraySize);
    
    createFlag(w) = false;
end

% SaveHDF(fileName, datasetName, input, inDims, startLoc, chunkSize, createFlag, arraySize)
% h5create(fileName, datasetName, [inDims(1:end-1), arraySize], 'Datatype', 'single', 'ChunkSize', [inDims(1:end-1), chunkSize]);
% h5write(fileName, datasetName, single(input), startLoc, inDims);





