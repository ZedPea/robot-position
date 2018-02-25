module Main
(
    main
)
where

import System.Random (getStdRandom, randomR)
import Control.Monad (replicateM)
import Data.List (tails)
import Text.Printf (printf)

import Paths_robot_position (getDataFileName)

type Weight = Double
type Input = Double
type Output = Double
type Error = Double
type Expected = Double

newtype Neuron = Neuron {
    weights :: [Weight]
}

main :: IO ()
main = do
    input <- map read . lines <$> (readFile =<< getDataFileName "src/exe/track.txt")
    neuron <- initNeuron
    -- can't predict the first result as no inputs for it so just skip it
    let trainedNeuron = trainNtimes 1000 neuron input (tail input)
    let results = think trainedNeuron (prepareInputs input 3)
    let output = concat $ zipWith formatOutput (tail input) results
    writeFile "output.txt" output

formatOutput :: Input -> Output -> String
formatOutput = printf "%f predicted to be %f\n"

think :: Neuron -> [[Input]] -> [Output]
think neuron = map (`neuronOutput` neuron)

trainNtimes :: Int -> Neuron -> [Input] -> [Expected] -> Neuron
trainNtimes n neuron input expectedOutputs = iterate train' neuron !! n
    where train' x = train x input expectedOutputs

initNeuron :: IO Neuron
initNeuron = Neuron <$> createRandomWeights 3 

createRandomWeights :: Int -> IO [Weight]
createRandomWeights amount = replicateM amount createRandomWeight
    where createRandomWeight = getStdRandom (randomR (-1, 1))

neuronOutput :: [Input] -> Neuron -> Output
neuronOutput inputs neuron = normalise weightedSum activationFunction
    where weightedSum = sum $ zipWith (*) inputs (weights neuron)

--makes it easy to change later
activationFunction :: Output -> Output
activationFunction = sigmoid

normalise :: a -> (a -> b) -> b
normalise = flip id

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp (-1 * x))

perceptron :: (Num p, Num a, Ord a) => a -> p
perceptron x
    | x > 0 = 1
    | otherwise = -1

adjustWeights :: [Input] -> Neuron -> Output -> Error -> Neuron
adjustWeights inputs neuron output errorVal = neuron { weights = newWeights }
    where newWeights = zipWith (+) adjustment (weights neuron)
          adjustment = map (errorVal * curveGradient *) inputs
          curveGradient = activationFunctionGradient output

activationFunctionGradient :: Output -> Output
activationFunctionGradient output = output * (1 - output)

calculateError :: Expected -> Output -> Error
calculateError = (-)

train :: Neuron -> [Input] -> [Expected] -> Neuron
train neuron inputs expectedOutputs = foldl predict neuron (zip windows expectedOutputs)
    -- we need an extra 0 for the amount of inputs - 1 we take in one step
    -- e.g. the first input is [0, 0, x1] if we have three inputs
    -- second input is [0, x1, x2]
    where windows = prepareInputs inputs numInputs 
          numInputs = length $ weights neuron

-- need to skip the last one - no value to predict
prepareInputs :: [Input] -> Int -> [[Input]]
prepareInputs inputs numInputs = init $ takeInputs numInputs inputs'
    where inputs' = replicate (numInputs - 1) 0 ++ inputs

--split our inputs into sliding windows of n size
takeInputs :: Int -> [Input] -> [[Input]]
takeInputs n inputs = filter (\x -> length x == n) . map (take n) $ tails inputs

predict :: Neuron -> ([Input], Output) -> Neuron
predict neuron (inputs, expectedOutput) = adjustWeights inputs neuron output errorVal
    where output = neuronOutput inputs neuron
          errorVal = calculateError expectedOutput output
