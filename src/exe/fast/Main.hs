module Main
(
    main
)
where

import System.Random (getStdRandom, randomR)
import Control.Monad (replicateM)
import Text.Printf (printf)
import Debug.Trace (trace)

import Numeric.LinearAlgebra 
    (Matrix, Vector, vector, fromRows, vjoin, subVector, size, cmap, tr, 
     sumElements, toList, (#>))

import Paths_robot_position (getDataFileName)

type Weight = Double
type Input = Double
type Output = Double
type Expected = Double
type Loss = Double

newtype Neuron = Neuron {
    weights :: Vector Weight
}

main :: IO ()
main = do
    input <- vector . map read . lines <$> (readFile =<< getDataFileName "src/exe/compact/track.txt")
    neuron <- Neuron <$> createRandomWeights 

    let windows = prepareInputs input
    -- can't predict the first result as no inputs for it so just skip it
        trainedNeuron = trainWhile neuron windows (subVector 0 (size input - 1) input)
        results = neuronOutput trainedNeuron windows
        output = formatOutput input results

    writeFile "output.txt" output

formatOutput :: Vector Input -> Vector Output -> String
formatOutput input' output = concat $
    zipWith (printf "%f predicted to be %f\n") (toList input) (toList output)
    where input = subVector 0 (size input' - 1) input'

numInputs :: Int
numInputs = 3

trainWhile :: Neuron -> Matrix Input -> Vector Expected -> Neuron
trainWhile neuron windows expected
    -- loss levels out at 1.08
    | loss < 1.09 = newNeuron
    | otherwise = trace ("Loss = " ++ show loss) (trainWhile newNeuron windows expected)
    where newNeuron = train neuron windows expected
          loss = calculateLoss expected output
          output = neuronOutput newNeuron windows

calculateLoss :: Vector Expected -> Vector Output -> Loss
calculateLoss expected output = 0.5 * sumElements (cmap (**2) (expected - output))

createRandomWeights :: IO (Vector Weight)
createRandomWeights = vector <$> replicateM numInputs (getStdRandom (randomR (-1, 1)))

train :: Neuron -> Matrix Input -> Vector Expected -> Neuron
train neuron windows expected = Neuron $ tr windows #> errorDerivative
    where outputs = neuronOutput neuron windows
          errorV = expected - outputs
          errorDerivative = errorV * cmap sigmoid' outputs

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-1 * x))

sigmoid' :: Num a => a -> a
sigmoid' x = x * (1 - x)

neuronOutput :: Neuron -> Matrix Input -> Vector Output
neuronOutput neuron windows = cmap sigmoid $ windows #> weights neuron

-- need to skip the last one - no value to predict
prepareInputs :: Vector Input -> Matrix Input
prepareInputs inputs = fromRows windows
    where inputs' = vjoin [vector $ replicate (numInputs - 1) 0, inputs]
          windows = map (\n -> subVector n numInputs inputs') [0 .. size inputs - 2]
