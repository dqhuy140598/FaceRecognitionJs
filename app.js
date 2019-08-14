require('dotenv').config();
const express = require('express');

const app = express()

const PORT = process.env.PORT || 3000

const path = require('path');


app.use(express.static(path.join(__dirname,'public')))

app.get('',(req,res) => {
    return res.send({
        message:'Hello Express'
    })    
})
app.get('/test',async (req,res) => {
    return res.sendFile(path.join(__dirname,'views/index.html'))    
})

app.get('/streamer',(req,res) => {
    return res.sendFile(path.join(__dirname,'views/minimalWebcam.html'))
})


app.listen(PORT,() => {
    console.log('Server is on port'+ PORT)
})