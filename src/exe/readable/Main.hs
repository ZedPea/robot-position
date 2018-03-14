module Main
(
    main
)
where

import System.Random (getStdRandom, randomR)
import Control.Monad (replicateM)
import Data.List (tails)
import Text.Printf (printf)
import Control.Concurrent (forkIO)
import Control.Concurrent.MVar (MVar, putMVar, newEmptyMVar, isEmptyMVar)
import Graphics.Rendering.Chart.Easy
import Graphics.Rendering.Chart.Backend.Cairo

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

    dieMVar <- newEmptyMVar

    forkIO $ exitWatcher dieMVar

    let windows = prepareInputs input
    -- can't predict the first result as no inputs for it so just skip it

    trainedNeuron <- trainWhile neuron windows (tail input) dieMVar

    let results = neuronOutput trainedNeuron windows
        output = concat $ zipWith (printf "%f predicted to be %f\n") (tail input) results

    putStrLn "Wrote predictions to output.txt"
    writeFile "output.txt" output

    putStrLn "Wrote graph to graph.png"
    writeGraph (tail input) results

    putStrLn "Bye."

writeGraph :: [Expected] -> [Output] -> IO ()
writeGraph expected results = toFile def "graph.png" $ do
    layout_title .= "Robot Position"
    plot (line "Actual" [zip ([1..] :: [Int]) expected])
    plot (line "Predicted" [zip [1..] results])

exitWatcher :: MVar Bool -> IO ()
exitWatcher dieMVar = do
    input <- getLine
    if input == "exit"
        then do
            putMVar dieMVar True
            return ()
        else exitWatcher dieMVar

numInputs :: Int
numInputs = 3

learningRate :: Double
learningRate = 0.01

trainWhile :: Neuron -> [[Input]] -> [Expected] -> MVar Bool -> IO Neuron
trainWhile neuron windows expected dieMVar = do
    die <- not <$> isEmptyMVar dieMVar
    if die
        then return neuron
        else do
            let newNeuron = train neuron windows expected
                output = neuronOutput neuron windows
                loss = 0.5 * sum (map (**2) $ zipWith (-) expected output)

            putStrLn $ "Loss = " ++ show loss

            trainWhile newNeuron windows expected dieMVar

createRandomWeights :: IO [Weight]
createRandomWeights = replicateM numInputs $ getStdRandom (randomR (-1, 1))

neuronOutput :: Neuron -> [[Input]] -> [Output]
neuronOutput = map . ((sigmoid . sum) .) . zipWith (*) . weights
    where sigmoid x = 1 / (1 + exp (-1 * x))

adjustWeights :: [Input] -> Output -> Error -> Neuron -> Neuron
adjustWeights inputs output errorVal = Neuron . zipWith (+) adjustment . weights
    where adjustment = map (learningRate * errorVal * sigmoid' output *) inputs
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
