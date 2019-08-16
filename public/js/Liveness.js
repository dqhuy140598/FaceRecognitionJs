const EAR_THRESHOLD = 0.3
const MAR_THRESHOLD = 0.3
const EAR_CONSECUTIVE_FRAMES = 1

const checkLiveNess = async function(leftEyePoints,rightEyePoints,mouthPoints){
    const {isClose:isLeftEyeClose,ear:leftEyeEar }= await checkEyeClose(leftEyePoints);
    const {isClose:isRightEyeClose,ear:rightEyeEar} = await checkEyeClose(rightEyePoints);
    //const {isOpen:isMouthOpen,mar:mouthMar} = await checkMouthOpen(mouthPoints);

    console.log('is left eye close: ',isLeftEyeClose);
    console.log('is right eye close: ',isRightEyeClose);
    console.log('left eye ear: ',leftEyeEar);
    console.log('right eye ear: ',rightEyeEar)

    if(isLeftEyeClose || isRightEyeClose){
         return true;
    }
    return false
}


const checkEyeClose = async function(eyePoints) {
    const tensorEyePoints = [];
    eyePoints.forEach(point => {
        const x = point.x;
        const y = point.y;
        const tensorPoint = tf.tensor1d([x,y]); 
        tensorEyePoints.push(tensorPoint)
    });
    const ear = await eyeAspectRatio(tensorEyePoints);
    return {isClose:ear<EAR_THRESHOLD,ear}
}

const eyeAspectRatio = async function(eyePoints) {
    const scalarWeights = tf.scalar(2.0)
    const component1 = tf.sub(eyePoints[0],eyePoints[3]);
    const component2 = tf.sub(eyePoints[1],eyePoints[5]);
    const component3 = tf.sub(eyePoints[2],eyePoints[4]);
    const ts = tf.add(tf.norm(component2),tf.norm(component3));
    const ms = tf.mul(scalarWeights,tf.norm(component1));
    const earTensor = tf.div(ts,ms);
    const ear = await earTensor.data()
    return ear[0]
}


const checkMouthOpen = async function(mouthPoints){
    const tensorMouthPoints = [];
    mouthPoints.forEach(point => {
        const x = point.x;
        const y = point.y;
        const tensorPoint = tf.tensor1d([x,y]); 
        tensorMouthPoints.push(tensorPoint)
    });

    const mar = await mouthAspectRatio(tensorMouthPoints);
    return {isOpen:mar<MAR_THRESHOLD,mar}
}

const mouthAspectRatio = async function(mouthPoints){
    const scalarWeights = tf.scalar(2.0)
    const component1 = tf.sub(mouthPoints[0],mouthPoints[4]);
    const component2 = tf.sub(mouthPoints[1],mouthPoints[7]);
    const component3 = tf.sub(mouthPoints[2],mouthPoints[6]);
    const component4 = tf.sub(mouthPoints[3],mouthPoints[5]);
    const ts1 = tf.add(tf.norm(component2),tf.norm(component3));
    const ts2 = tf.add(tf.norm(component4),ts1)
    const ms = tf.mul(scalarWeights,tf.norm(component1));
    const marTensor = tf.div(ts2,ms);
    const mar = await marTensor.data()
    return mar[0]
}