import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

val height = 28
val width = 28
val channels = 1
val numClasses = 10

val conf = MultiLayerConfiguration.Builder()
            .inputType(InputType.convolutional(height, width, channels))
            .convolutionMode(ConvolutionMode.Same)
            .list()
            .layer(0, ConvolutionLayer.Builder(5, 5)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.RELU)
                    .build())
            .layer(1, SubsamplingLayer.Builder(PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(2, ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1)
                    .nOut(50)
                    .activation(Activation.RELU)
                    .build())
            .layer(3, SubsamplingLayer.Builder(PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(4, DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500).build())
            .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(numClasses)
                    .activation(Activation.SOFTMAX)
                    .build())
            .backprop(true).pretrain(false)
            .build()

val net = MultiLayerNetwork(conf)
net.init()
