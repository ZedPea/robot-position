import System.Random (getStdRandom, randomR)
import Control.Monad (replicateM, ap)
import Data.List (tails)

main = do
    input <- map read . lines <$> readFile "track.txt" :: IO [Double]
    neuron <- replicateM 3 (getStdRandom (randomR (-1, 1)))
    writeFile "output.txt" . concat . zipWith (\x y -> show x ++ " predicted to be " ++ show y) (tail input) . neuronOutput (trainWhile neuron (prepareInputs input) (tail input)) $ prepareInputs input

trainWhile neuron windows expected | loss < 1.1 = train neuron windows expected | otherwise = trainWhile (train neuron windows expected) windows expected
    where loss = 0.5 * sum (map (**2) . zipWith (-) expected $ neuronOutput neuron windows)

neuronOutput = map . (((1 /) . (1 +) . exp . negate . sum) .) . zipWith (*)

adjustWeights = ((zipWith (+) .) .) . flip (flip . ((map . (*)) .) . (*) . ap (*) (1 -))

predict n (inputs, expected') = adjustWeights inputs output (expected' - output) n
    where [output] = neuronOutput n [inputs]

train neuron windows expected = foldl predict neuron (zip windows expected)

prepareInputs = init . filter ((3 ==) . length) . map (take 3) . tails . (replicate 2 0 ++)
