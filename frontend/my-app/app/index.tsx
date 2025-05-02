import { Text, View } from "react-native";
import PredictionForm from "./components/PredictionForm";

export default function Index() {
  return (
    <View
      style={{
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <h1 style={{fontSize: 'larger'}}>Cyber Attack Prediction</h1>
      <PredictionForm />
    </View>
  );
}
