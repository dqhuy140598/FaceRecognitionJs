const widthRatioThresh  = 0.2031
const heightRatioThresh = 0.2916
const areaRatioThresh = 0.0592

const checkSize = function(faceBoundingBox,{imageWidth,imageHeight}){
    const {x,y,width:faceWidth,height:faceHeight} = faceBoundingBox;
    if(faceWidth/imageWidth < widthRatioThresh || faceHeight/imageHeight < heightRatioThresh){
        return false;
    }
    const faceArea = faceWidth * faceHeight;
    const imageArea = imageWidth * imageHeight;
    if(faceArea/imageArea < areaRatioThresh){
        return false
    }
    return true
}