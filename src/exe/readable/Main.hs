module Main
(
    main
)
where

import System.Random (getStdRandom, randomR)
import Control.Monad (replicateM)
import Data.List (tails)
import Text.Printf (printf)
import Debug.Trace (trace)

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
    input <- map read . lines <$> (readFile =<< getDataFileName "src/exe/compact/track.txt")
    neuron <- Neuron <$> createRandomWeights 

    let windows = prepareInputs input
    -- can't predict the first result as no inputs for it so just skip it
        trainedNeuron = trainWhile neuron windows (tail input)
        results = neuronOutput trainedNeuron windows
        output = concat $ zipWith (printf "%f predicted to be %f\n") (tail input) results

    writeFile "output.txt" output

numInputs :: Int
numInputs = 7

trainWhile :: Neuron -> [[Input]] -> [Expected] -> Neuron
trainWhile neuron windows expected
    | loss < 0.5 = newNeuron
    | otherwise = trace ("Loss = " ++ show loss) (trainWhile newNeuron windows expected)
    where newNeuron = train neuron windows expected
          loss = 0.5 * sum (map (**2) $ zipWith (-) expected output)
          output = neuronOutput neuron windows

createRandomWeights :: IO [Weight]
createRandomWeights = replicateM numInputs $ getStdRandom (randomR (-1, 1))

neuronOutput :: Neuron -> [[Input]] -> [Output]
neuronOutput = map . ((sigmoid . sum) .) . zipWith (*) . weights
    where sigmoid x = 1 / (1 + exp (-1 * x))

adjustWeights :: [Input] -> Output -> Error -> Neuron -> Neuron
adjustWeights inputs output errorVal = Neuron . zipWith (+) adjustment . weights
    where adjustment = map (errorVal * sigmoid' output *) inputs
          sigmoid' x = x * (1 - x)

train :: Neuron -> [[Input]] -> [Expected] -> Neuron
train neuron windows expected = foldl predict neuron (zip windows expected)
    where predict n (inputs, expected') =
            let [output] = neuronOutput n [inputs]
            in  adjustWeights inputs output (expected' - output) n

-- need to skip the last one - no value to predict
prepareInputs :: [Input] -> [[Input]]
prepareInputs = init . takeInputs . (replicate (numInputs - 1) 0 ++)

--split our inputs into sliding windows of n size
takeInputs :: [Input] -> [[Input]]
takeInputs inputs = filter (\x -> length x == n) . map (take n) $ tails inputs
    where n = numInputs
